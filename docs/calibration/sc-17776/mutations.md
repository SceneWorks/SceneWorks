# sc-17776 hand mutations

Seven one-variable edits on top of inference `fbb00d6b4147bd220fe324050534708c12fb022d`, applied to a
detached worktree and never committed. Historical commits change more than one thing at a time —
sc-17775 §4.1 rows A and D are both contaminated for exactly that reason — so the discriminating
claims in `docs/calibration-invalidation-unit-sc-17776.md` rest on these instead.

Each is reproduced by the `sed`/append below, from a clean checkout of `fbb00d6b`. The measurement
driver applies them with `git apply` and resets the worktree between cases.

| # | Class | What it changes | What a sound unit should say |
| --- | --- | --- | --- |
| `M-rms` | **known-live** (the story's named mutation) | `const RMS_EPS: f64` in `candle-gen-flux2/src/transformer.rs`, `1e-5` → `2e-5` | **DIFFER** — production numerics on the measured path |
| `M-devfp` | positive control | `flux2_dev`'s `CALIBRATION_FINGERPRINT`, `…blocks-v2` → `…blocks-v3` | **DIFFER** — the provider declaring its own calibration void |
| `M-klein` | crate-mate, inert for dev | `KLEIN_CALIBRATION_FINGERPRINT`, `…abi-v2` → `…abi-v3` | ideally **identical** — `flux2_dev` cannot execute it |
| `M-editprov` | crate-mate in a module dev never enters | the reference-grid offset in `edit_provider.rs`, `10 + 10 * i` → `12 + 10 * i` | ideally **identical** — `Flux2Edit` is a bespoke klein-edit provider, unregistered and unreachable from the dev txt2img path |
| `M-cfgtest` | `#[cfg(test)]` inside the audited crate | appends a `#[cfg(test)] mod` with one trivial test to `candle-gen-flux2/src/lib.rs` | ideally **identical** — no production codegen |
| `M-gencore` | `gen-core` addition nothing references | appends an unreferenced `pub fn` to `crates/contracts/gen-core/src/lib.rs` | **identical** — the link-time-DCE control |
| `M-safety` | **known-live**, admission path | `validate_context`'s guard, `context.geometry.batch != 1` → `batch > 2` | **DIFFER** — it admits a two-image batch into a ladder whose own message says single-image only |

`M-rms`, `M-devfp` and `M-safety` are three independent live changes, deliberately on three different
surfaces — the transformer numerics, the calibration identity, and the admission guard — so that a
unit which happens to link one of them cannot pass by luck.

The **two-sided control is per unit**, because the units differ in what they can see. `M-gencore` is
the only row expecting `identical` on every unit, so it catches a stuck-DIFFER instrument anywhere.
On the far side it depends which unit is being checked: the code units are held by `M-rms`, the
ladder-scoped digest by `M-devfp` and `M-safety` (it is blind to `M-rms`), and the behaviour witness
by `M-devfp` alone (it is blind to both `M-rms` and `M-safety`). Every unit reported in
`docs/calibration-invalidation-unit-sc-17776.md` is exercised in both directions; none of them is
absolving or refusing everything.

`M-klein`, `M-editprov` and `M-cfgtest` discriminate nothing on their own — they are the
over-triggers under test, and a broken unit returns `DIFFER` for them too. Their value is in which
unit absolves them, not in the verdict itself.

## Reproducing the edits

```bash
cd <worktree at fbb00d6b>

# M-rms
sed -i 's/^const RMS_EPS: f64 = 1e-5;/const RMS_EPS: f64 = 2e-5;/' \
  crates/media/candle-gen/candle-gen-flux2/src/transformer.rs

# M-devfp
sed -i 's/flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v2/flux2-dev-cuda-staged-host-full-edge-decode-bounded-attention-device-format-blocks-v3/' \
  crates/media/candle-gen/candle-gen-flux2/src/memory_strategy.rs

# M-klein
sed -i 's/flux2-klein-cuda-shared-ladder-provider-abi-v2/flux2-klein-cuda-shared-ladder-provider-abi-v3/' \
  crates/media/candle-gen/candle-gen-flux2/src/memory_strategy.rs

# M-editprov
sed -i 's/                10 + 10 \* i as i64,/                12 + 10 * i as i64,/' \
  crates/media/candle-gen/candle-gen-flux2/src/edit_provider.rs

# M-cfgtest
cat >> crates/media/candle-gen/candle-gen-flux2/src/lib.rs <<'EOF'

#[cfg(test)]
mod sc_17776_probe_tests {
    #[test]
    fn sc_17776_probe_trivial() {
        assert_eq!(2 + 2, 4);
    }
}
EOF

# M-gencore
cat >> crates/contracts/gen-core/src/lib.rs <<'EOF'

/// sc-17776 probe: an item nothing in FLUX.2's closure references, so link-time DCE should drop it.
pub fn sc_17776_unreferenced_probe(x: u64) -> u64 {
    x.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31)
}
EOF

# M-safety (line number is from fbb00d6b; the guard is inside `pub fn validate_context`)
sed -i '365s/if context.geometry.batch != 1 {/if context.geometry.batch > 2 {/' \
  crates/media/candle-gen/candle-gen-flux2/src/memory_strategy.rs
```

`M-klein`, `M-devfp` and `M-safety` all edit `memory_strategy.rs` and are applied separately, never
together.

**Comparing with sc-17775.** `M-klein` and `M-devfp` reproduce that survey's M1 and M3 digests
exactly (`73fe78d5…` and `eb9a6ead…`), and `M-gencore` reproduces its M4 (`fee1c2de…`, i.e. the
unmutated binary). `M-cfgtest` does **not** reproduce its M2 (`0b101ed9…` here against `750a0cfe…`
there) and should not: both append a trivial `#[cfg(test)] mod`, but not the same text, and the
digest is of the compiled result rather than of the description.
