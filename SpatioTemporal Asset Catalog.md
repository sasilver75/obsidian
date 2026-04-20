---
aliases:
  - STAC
tags:
---
A JSON metadata standard for discovering [[Raster]] datasets.
- Tool to create Catalogs

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


![[Pasted image 20260419225513.png]]

![[Pasted image 20260419225548.png]]
![[Pasted image 20260419225709.png]]
Above: If you want to query only the building footprints in Austin, you can do that from within [[DuckDB]], here reading from an [[Amazon S3|S3]] URL that [[Overture Maps Foundation|Overture]] maps hosts. This is without needing a ton of money/compute.

![[Pasted image 20260419225657.png]]
Same thing with [[Zarr]], requesting 2 meter resolution temperature, with the data streaming into the bucket every hour or so, I think.

![[Pasted image 20260419225747.png]]
![[Pasted image 20260419225922.png]]
Planetary compute stack API, looking [[Sentinel|Sentinel-2]] data with a bounding box over Austin in a certain date range, looking for the best image with a certain cloud coverage.
- This returns the full satellite image intersecting your [[Area of Interest|AOI]], and you still need to clip it.
He made a false color visualization where he swapped out the blue band with the [[Near Infrared|NIR]] band, which is good for vegetation analysis... [[Normalized Difference Vegetation Index|NDVI]] is the near-infrared minus the red band, normalized; bright green is healthy vegetation.

![[Pasted image 20260419230142.png]]
[[National Agriculture Imagery Program]] (NAIP) flies the US with really high resolution (~1m, sub-meter) every two years or so.





