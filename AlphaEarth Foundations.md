---
aliases:
  - AlphaEarth
  - AEF
  - Google Satellite Embeddings
---


Blog: https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/
Paper: https://arxiv.org/abs/2507.22291 
September 8, 2025

A geospatial foundation model from [[DeepMind|Google DeepMind]] that produces "embedding fields," dense 64-byte vectors at 10m square resolution covering Earth's land surfaced, released annually from 2017-2014 on [[Google Earth Engine]] (GEE).
- Not open-weights; you can't fine-tune the model itself, unlike something like [[Prithvi 2.0]]
- Produced the [[AlphaEarth Foundations|Google Satellite Embeddings]] dataset, a global analysis- ready collection of learned geospatial embeddings ([LINK](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL))

Key innovations:
- Multi-source fusion
	- Optical ([[Sentinel|Sentinel-2]], [[Landsat]])
	- Radar ([[Sentinel|Sentinel-1]], [[PALSAR|PALSAR-2]])
	- LiDAR ([[Global Ecosystem Dynamics Investigation|GEDI]])
	- Climate (ERA5, GRACE)
	- Elevation ([[Copernicus DEM|GLO-30]])
	- Annotations (NLCD, Wikipedia + GBIF species records)
- Continuous time modeling: First [[Remote Sensing|EO]] embedding approach to treat time as a continuous variable, not discrete snapshots. You can query embeddings for any date range, even interpolating/extrapolating beyond observed data.
- Space Time Precision (STP) encoder: Three simultaneous pathways at different resolutions:
	- 1/16 L spatial self-attention ([[Vision Transformer|ViT]]-style)
	- 1/8 L time-axial self-attention
	- 1/2 L convolutional precision path
- Interleaved with learned Laplacian pyramid exchanges to main both global context and local precision.
- Batch uniformity objective: Forces embeddings onto the unit sphere (S^63), preventing representational collapse.


Performance:
- ~23.9% error reduction over next-best method across 15 evaluations
- Outperforms [[Clay]], [[Prithvi]], [[SatCLIP]], and designed features ([[Continuous Change Detection and Classification|CCDC]], [[Multi-task Observation using Satellite Imagery & Kitchen Sinks|MOSAIKS]]) consistently across all tasks; the ==first approach where no single competitor wins anywhere==.
- Only 64 bytes per embedding vs. 16x more for next-most-compact earned method
- ~==480M parameter model== (smaller variant chosen for inference efficiency)








