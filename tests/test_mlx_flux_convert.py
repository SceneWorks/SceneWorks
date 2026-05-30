"""Unit tests for the FLUX.2-klein single-file -> diffusers transformer converter
(sc-2235). Exercises the framework-agnostic key plan with numpy, so it runs in
the main worker venv without mlx. The real tensor conversion was validated on
hardware in sc-2220; these tests are regression guards on the mapping, the
double-block qkv row-split, and the load-bearing norm_out scale/shift swap.
"""
from __future__ import annotations

import numpy as np

from scene_worker.mlx_flux_convert import build_target_state_dict

N_DOUBLE = 8
N_SINGLE = 24
D = 8  # tiny stand-in hidden dim


def _np_chunk3(t):
    return tuple(np.split(t, 3, axis=0))


def _np_swap_halves(t):
    shift, scale = np.split(t, 2, axis=0)
    return np.concatenate([scale, shift], axis=0)


def _synthetic_original_state_dict() -> dict:
    """All original (BFL/ComfyUI-convention) FLUX.2-klein transformer keys with
    plausible-but-tiny shapes."""
    sd: dict = {}
    sd["img_in.weight"] = np.random.rand(D, 4).astype(np.float32)
    sd["txt_in.weight"] = np.random.rand(D, 3 * D).astype(np.float32)
    sd["time_in.in_layer.weight"] = np.random.rand(D, 4).astype(np.float32)
    sd["time_in.out_layer.weight"] = np.random.rand(D, D).astype(np.float32)
    sd["double_stream_modulation_img.lin.weight"] = np.random.rand(6 * D, D).astype(np.float32)
    sd["double_stream_modulation_txt.lin.weight"] = np.random.rand(6 * D, D).astype(np.float32)
    sd["single_stream_modulation.lin.weight"] = np.random.rand(3 * D, D).astype(np.float32)
    sd["final_layer.linear.weight"] = np.random.rand(4, D).astype(np.float32)
    sd["final_layer.adaLN_modulation.1.weight"] = np.random.rand(2 * D, D).astype(np.float32)
    for i in range(N_DOUBLE):
        s = f"double_blocks.{i}"
        for stream in ("img", "txt"):
            sd[f"{s}.{stream}_attn.qkv.weight"] = np.random.rand(3 * D, D).astype(np.float32)
            sd[f"{s}.{stream}_attn.proj.weight"] = np.random.rand(D, D).astype(np.float32)
            sd[f"{s}.{stream}_attn.norm.query_norm.weight"] = np.random.rand(D).astype(np.float32)
            sd[f"{s}.{stream}_attn.norm.key_norm.weight"] = np.random.rand(D).astype(np.float32)
            sd[f"{s}.{stream}_mlp.0.weight"] = np.random.rand(3 * D, D).astype(np.float32)
            sd[f"{s}.{stream}_mlp.2.weight"] = np.random.rand(D, 3 * D).astype(np.float32)
    for i in range(N_SINGLE):
        s = f"single_blocks.{i}"
        sd[f"{s}.linear1.weight"] = np.random.rand(9 * D, D).astype(np.float32)
        sd[f"{s}.linear2.weight"] = np.random.rand(D, 4 * D).astype(np.float32)
        sd[f"{s}.norm.query_norm.weight"] = np.random.rand(D).astype(np.float32)
        sd[f"{s}.norm.key_norm.weight"] = np.random.rand(D).astype(np.float32)
    return sd


def test_key_plan_matches_diffusers_klein_9b_cardinality():
    src = _synthetic_original_state_dict()
    out = build_target_state_dict(src, chunk3=_np_chunk3, swap_halves=_np_swap_halves)
    # 9 top-level + 16 per double block (6 qkv splits + 10 renames) + 4 per single block.
    assert len(out) == 9 + 16 * N_DOUBLE + 4 * N_SINGLE == 233
    assert len(set(out)) == len(out)  # no collisions


def test_top_level_and_block_keys_present():
    src = _synthetic_original_state_dict()
    out = build_target_state_dict(src, chunk3=_np_chunk3, swap_halves=_np_swap_halves)
    for key in (
        "x_embedder.weight",
        "context_embedder.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.7.attn.add_v_proj.weight",
        "transformer_blocks.3.attn.to_out.0.weight",
        "transformer_blocks.3.ff_context.linear_out.weight",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
        "single_transformer_blocks.23.attn.norm_k.weight",
    ):
        assert key in out, key
    # original-convention keys must NOT leak through
    assert not any(k.startswith(("double_blocks.", "single_blocks.", "img_in", "txt_in")) for k in out)


def test_double_block_qkv_row_split_order():
    src = _synthetic_original_state_dict()
    out = build_target_state_dict(src, chunk3=_np_chunk3, swap_halves=_np_swap_halves)
    qkv = src["double_blocks.0.img_attn.qkv.weight"]
    np.testing.assert_array_equal(out["transformer_blocks.0.attn.to_q.weight"], qkv[:D])
    np.testing.assert_array_equal(out["transformer_blocks.0.attn.to_k.weight"], qkv[D : 2 * D])
    np.testing.assert_array_equal(out["transformer_blocks.0.attn.to_v.weight"], qkv[2 * D :])
    txt = src["double_blocks.0.txt_attn.qkv.weight"]
    np.testing.assert_array_equal(out["transformer_blocks.0.attn.add_q_proj.weight"], txt[:D])
    np.testing.assert_array_equal(out["transformer_blocks.0.attn.add_v_proj.weight"], txt[2 * D :])


def test_norm_out_scale_shift_swap():
    """The load-bearing transform: BFL (shift, scale) -> diffusers (scale, shift)."""
    src = _synthetic_original_state_dict()
    out = build_target_state_dict(src, chunk3=_np_chunk3, swap_halves=_np_swap_halves)
    adaln = src["final_layer.adaLN_modulation.1.weight"]
    shift, scale = adaln[:D], adaln[D:]
    result = out["norm_out.linear.weight"]
    np.testing.assert_array_equal(result[:D], scale)  # scale now first
    np.testing.assert_array_equal(result[D:], shift)  # shift now second
    assert not np.array_equal(result, adaln)  # genuinely reordered
