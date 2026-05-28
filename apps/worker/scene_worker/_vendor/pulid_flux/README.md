# PuLID-FLUX (vendored)

Vendored copy of [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) — Apache-2.0
(see `LICENSE`). Powers `PuLIDFluxAdapter` (`apps/worker/scene_worker/pulid_flux_adapter.py`)
for FLUX face-identity character generation (sc-2012, epic 2003).

## Provenance

- Upstream commit: `1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa` ("support FLUX.1-Krea-dev")
- Subdirs copied verbatim: `pulid/`, `flux/`, `eva_clip/`
- Dropped: `pulid/pipeline.py` and `pulid/pipeline_v1_1.py` (PuLID v1.1 / SDXL paths
  unused by sc-2012), `pulid/encoders.py` (only referenced by the dropped v1.1 path),
  the upstream `app.py` / `app_flux.py` gradio demos and `requirements*.txt`.

## Local patches

Marked with `# PATCH (SceneWorks):` so they're greppable when re-vendoring upstream.

1. **`flux/math.py::rope`** — MPS has no `float64` kernel. Switch to `float32` on
   MPS, keep `float64` on CUDA/CPU. (sc-2012 spike finding.)
2. **`flux/util.py::load_flow_model` + `load_ae`** — Drop the `local_dir='models'`
   kwarg on `hf_hub_download`. Upstream uses it so the demo's `models/` dir gets
   populated; the worker process can't rely on a writable `models/` alongside cwd.
   `hf_hub_download` without `local_dir` returns the shared HF cache path directly
   (zero extra disk).
3. **`pulid/pipeline_flux.py::PuLIDPipeline.__init__`** — The upstream
   `snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')`
   call and the hard-coded `FaceAnalysis(root='.')` / `'models/antelopev2/glintr100.onnx'`
   paths are replaced with a `PULID_FLUX_INSIGHTFACE_ROOT` env var (defaulting to
   `~/.insightface`). The worker pre-provisions the 5 antelopev2 ONNX files via the
   shared sc-2009 InstantID helper (`_ensure_antelopev2`); both adapters use the
   same insightface root, so the pack downloads once.
4. **`pulid/pipeline_flux.py::PuLIDPipeline.load_pretrain`** — Drop
   `local_dir='models'` here too; the PuLID-FLUX adapter weights live in the HF
   cache.

## Why vendored, not pip-installed

PuLID-FLUX is not packaged on PyPI; upstream is a script-style repo. Vendoring
gives us a deterministic version pin and carries the four patches above. The
runtime extras (`requirements-pulid-flux.txt`) are pip-installed normally.
