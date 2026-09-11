# StarVector-1B Vectorization Guide

StarVector-1B converts one raster image into SVG source. Choose a clear source with a distinct
silhouette, limited visual clutter, and enough contrast to separate the subject from its background.

This checkpoint is image-conditioned only. SceneWorks does not send free-form text guidance to it,
and the catalog intentionally advertises `image_to_svg` without `text_to_svg`.

The native provider reads the installed snapshot directly. If Model Manager reports the model as
missing or incomplete, download or repair it before submitting a vectorization job.

Source: [StarVector-1B image-to-SVG model card](https://huggingface.co/starvector/starvector-1b-im2svg)
