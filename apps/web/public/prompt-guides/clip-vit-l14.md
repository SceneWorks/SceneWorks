# CLIP ViT-L/14 (image embedder)

CLIP ViT-L/14 is OpenAI's image encoder. SceneWorks uses it as a small utility dependency — **it does not generate images or video, and there is nothing to prompt.** Dataset Doctor embeds each training image with it to measure dataset quality: near-duplicate detection, diversity/coverage, and caption↔image alignment.

## Installation

The native worker (MLX on macOS, candle on Windows/CUDA) resolves this model from the shared Hugging Face cache and does **not** auto-download it. Install it once from the **Models** screen; it downloads into the shared Hugging Face cache, so other tools reuse it.

If Dataset Doctor reports that the "CLIP embedder model path snapshot is not cached," this is the model to install.

## Practical Notes

There are no settings — the embedder runs automatically as part of dataset analysis. Once it is installed, re-run the dataset analysis and the embedding-based findings (duplicates, diversity, alignment) will populate.
