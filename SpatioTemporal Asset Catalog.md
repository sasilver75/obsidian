---
aliases:
  - STAC
tags:
  - Catalog
---
A JSON metadata standard for discovering [[Raster]] datasets.

Commonly paired with [[Cloud-Optimized GeoTIFF]]s (COGs); ==STAC+COG is the standard stack for satellite imagery.==
- STAC: Tells you what exists and where the files are
- COG: Lets you efficient read just the spatial subset you need

An open specification for describing geospatial assets in a consistent, searchable way.
Problem it solves: Every satellite data provider had their own metadata format and discovery mechanism, and STAC standardizes both:

Four components:
1. Item: Describes one asset (one Landsat scene, one Sentinel-2 tile; a [[GeoJSON]] feature).
2. Collection: Groups related items (all Sentinel-2 L2A scenes).
3. Catalog: A top-level container linking to collections.
4. API: Standard REST search endpoint; filter by bbox, datetime, cloud cover, collections.






