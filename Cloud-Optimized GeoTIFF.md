---
aliases:
  - COG
---
See also: [[Tagged Image File Format|TIFF]], [[GeoTIFF]]

References:
- [{CNG 2025} Is COG Scalable - Jeff Albrecht](https://youtu.be/-9tNOBRidVA)
- [Cloud Optimized Geospatial Formats Guide](https://guide.cloudnativegeo.org/cloud-optimized-geotiffs/intro.html) (Advanced notes note shown here)
- [Cloud Optimized GeoTIFF](https://cogeo.org/) website

> A CLoud-Optimized GeoTIFF (COG) is a [[Raster]] format and variant of the [[Tagged Image File Format|TIFF]] image format that specifies a particular layout of internal data in the [[GeoTIFF]] specification to allow for optimized (subsetted or aggregated) access over a network via [[HTTP Range Request]]s for display or data reading. All COG files are valid GeoTIFF files, but not all GeoTIFF files are valid COG files. They key components that differ beteween GeoTIFF and COG Are overviews and internal tiling.


A [[GeoTIFF]] with a specific internal organization designed for efficient remote access.
- Often paired with [[SpatioTemporal Asset Catalog|STAC]]s

A [[Cloud-Native Geospatial]] ==[[Raster]] format== designed for efficient [[HTTP Range Request]]s -- you can fetch just a tile-sized chunk of a massive satellite image without downloading the whole thing.

This is essential for serving satellite imagery.
- Tools like [[TiTiler]] (a [[Tile Server]]) consume COGs directly.

Two key internal features:
- ==Internal Tiling==: Instead of storing rows left-to-right, top-to-bottom (strip layout),  the image is divided into tiles (e.g. 256x256 or 512x512 pixels). You can read one tile without touching the rest of the file.
- ==Overview Levels== (Pyramids): Lower-resolution versions of the image are embedded in the same file; When you request a spatial window at a given resolution, the COG reader picks the overview level closest to your target resolution and reads from that.
	- Each zoom level is typically 2x ***downsampled*** from the previous (a factor of 4 in pixel count).
		- There are [many resampling algorithms](https://gdal.org/en/stable/programs/gdal_translate.html#cmdoption-gdal_translate-r) for generating overviews, several options should be compared before deciding which resampling method to apply.
	- The overviews are stored inside the same file; because of COG's ==header-first layout==, the reader knows exactly where each overview level lives before reading any pixel data.
	- ==One HTTP request gets you to the index, and then targeted range requests get you exactly the tiles you need.==


Request: "Give me zoom level 8 view of this scene."
- Read header: Find overview 3 offset
- Read only the specific tiles within overview 3 that intersect the viewport


> There is a tradeoff between storing lots of data in one GeoTIFF and storing less data in many GeoTIFFs. The larger a single file, the larger the GeoTIFF header, and multiple requests may be required just to read the spatial index (header) before data retrieval.
> The opposite problem occurs if you make too many small files; it can take many reads to retrieve data, and when rendering a combined visualization can greatly impact load time,
> The current recommendation is to meet somewhere in the middle, a moderate amount of medium files; 256x256 or 512x512 dimensions of data are recommended.
> "Dimensions" is the term for the number of bands stored in a GeoTIFF


![[Pasted image 20260419015503.png]]

![[Pasted image 20260419110130.png]]

![[Pasted image 20260419015535.png]]



![[Pasted image 20260423004238.png]]Left would be GeoTiff-type files