# StarVector terminal campaign

This is the single post-permanent-pin, dispatch-only SC-22261 campaign profile.
It never downloads snapshots, package wheels, or LPIPS weights: the selected
runner must already provide an immutable inference checkout, both model
inventories, an exact metric environment, and a production `vector_generate`
route executable. The controller rejects any missing or changed identity.

The workflow executes exactly four serial tuples: MLX 1B, MLX 8B, Candle/CUDA
1B, then Candle/CUDA 8B. Each must emit all 120 ordered image-quality cases and
20 deterministic parity cases. The last tuple also emits the one 200-case
hostile sanitizer suite and the one 60-case raster-to-vector prompt suite.
Every tuple artifact uploads even on a failed route; the final CPU-only seal
step accepts a receipt only after all raw evidence, source/output/transcript
digests, metrics lock, model inventories, and same campaign identifier agree.

The checked-in metric lock fixes white-composited 512x512 sRGB8 rendering,
scikit-image SSIM parameters, official LPIPS Alex evaluation parameters, and
the separately hashed LPIPS linear and AlexNet trunk weights. SceneWorks owns
the product route, sanitizer, rasterizer, asset publication and artifact
materialization; inference owns the terminal receipt schema and validator.
