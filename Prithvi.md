2023 [[Geospatial Embedding|Geospatial Foundation Model]] from NASA + IBM, with open weights on HF.
- Named after the Sanskrit word for Earth.

Uses a [[Masked Autoencoder]] with [[Vision Transformer|ViT]] backbone
- Pretraining objective: mask random patches of imagery, reconstruct them
- Same core idea as [[Clay]] but different training data and design choices

Trained on [[HLS]] (Harmonized [[Landsat]] [[Sentinel|Sentinel-2]]), NASA's analysis ready [[Surface Reflectance]] product that normalizes. with CONUS-focused training and ~4.2M image chips (compared to Clay's ~70M, or [[AlphaEarth Foundations|AlphaEarth]]'s ~1B observations).