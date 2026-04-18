See also: [[Tagged Image File Format|TIFF]], [[Cloud-Optimized GeoTIFF]] ("COG")

A [[Tagged Image File Format|TIFF]] with ==additional tags that encode geospatial metadata==:
- Projection/[[Coordinate Reference System|CRS]]: What coordinate system the image is in
- Geotransform: The mapping from pixel coordinates to geographic coordinates (origin, pixel size, rotation)
- Datum/Ellipsoid: A precise definition of the spatial reference

==So a GeoTIFF is just a regular TIFF that knows where on Earth it lives.==
- If you open it in a TIFF viewer, you just see an image.
- If you open it in [[Geospatial Data Abstraction Library|GDAL]]/[[QGIS]]/[[rasterio]], you also get the spatial context.


