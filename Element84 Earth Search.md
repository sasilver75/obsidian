---
aliases:
  - Element84
---

Compare with:
- [[Microsoft Planetary Computer]] (MPC) 
- [[Google Earth Engine]]


Earth Search is a ==[[SpatioTemporal Asset Catalog|STAC]]-compliant search and discovery API for [[Registry of Open Data on AWS]]==, allowing users to quickly and easily interact with their data of interest.
- The main onramp for accessing [[Sentinel|Sentinel-2]] [[Cloud-Optimized GeoTIFF|COG]]s on AWS Without going through [[Google Earth Engine|GEE]].
- STAC-compliant API endpoint that you can query by bbox, date range, cloud cover, dataset, etc, returning STAC items with direct S3 URLs to the actual data.
- Free to search; data transfer costs apply if you're not querying from within AWS.
- It's essentially a discovery layer that you can use to find what imagery exists for your areas of interest, then stream or download just the bands/chips you need directly from S3, rather than downloading entire scenes.

Provides:
- A [[SpatioTemporal Asset Catalog|STAC]]-compliant search and discovery API
- A web console for browsing/visualizing imagery
- A dashboard for monitoring what data is being processed/ingested into the API

Datasets available:
- [[Sentinel|Sentinel-1]] ([[Synthetic Aperture Radar|SAR]])
- [[Sentinel|Sentinel-2]] ([[Multispectral]], entire catalog as [[Cloud-Optimized GeoTIFF|COG]]s)
- [[Landsat]] Collection 2 (entire USSGS catalog)
- [[Copernicus DEM]]
- [[National Agriculture Imagery Program|NAIP]]

via the endpoint: `https://earth-search.aws.element84.com/v1`





