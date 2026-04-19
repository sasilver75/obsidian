---
aliases:
  - Radiant Earth
---

A nonprofit focused on open ML training data for [[Remote Sensing|Earth Observation]].
They're deeply embedded into the [[Cloud-Native Geospatial]] ecosystem ([[SpatioTemporal Asset Catalog|STAC]], [[Cloud-Optimized GeoTIFF|COG]]s, open data, humanitarian applications). 

Their main product is "[[Radiant MLHub]]", an open repository of labeled geoespatial training datasets for tasks like:
- Crop type mapping
- Flood extent detection
- Building footprint extraction
- Land cover classification
Think: "[[HuggingFace]] for satellite imagery training datasets, with a STAC API."

Why they matter:
- Getting labeled training data is the hardest part of EO ML. 
- Radiant earth curates, standardizes, and hosts these datasets openly so that researchers don't have to re-annotate the same imagery.
- All datasets are served by [[SpatioTemporal Asset Catalog|STAC]], to whose spec they were major contributors.

Key datasets:
- LandCoverNet: Global land cover labels
- CV4A Kenya Crop Type: Used in their annual competition
- Various flood/disaster response datasets from Sentinel-1/2

