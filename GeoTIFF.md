See also: [[Tagged Image File Format|TIFF]], [[Cloud-Optimized GeoTIFF]] ("COG")

A [[Tagged Image File Format|TIFF]] with ==additional tags that encode geospatial metadata==:
- Projection/[[Coordinate Reference System|CRS]]: What coordinate system the image is in
- Geotransform: The mapping from pixel coordinates to geographic coordinates (origin, pixel size, rotation)
- Datum/Ellipsoid: A precise definition of the spatial reference

==So a GeoTIFF is just a regular TIFF that knows where on Earth it lives.==
- If you open it in a TIFF viewer, you just see an image.
- If you open it in [[Geospatial Data Abstraction Library|GDAL]]/[[QGIS]]/[[rasterio]], you also get the spatial context.


A [[Tagged Image File Format|TIFF]] is a file format built around **tags**, key-value pairs stored in a header that describes the image. Standard tags cover things like image width, height, color depth, and compression. ==GeoTIFF simply adds more tags.==

The geospatial tags that GEOTiff adds:
- GeoKeyDirectoryTag: The core tag, containing of a set of "GeoKeys" that specify the [[Coordinate Reference System|CRS]] (whether it's a projected [[Coordinate Reference System|CRS]] (e.g. [[Universal Transverse Mercator|UTM]]) or a geographic CRS (lat/lon), specific [[EPSG Code|EPSG]] code, datum and ellipsoid.
- ModelPixelScaleTag: Pixel size in ground units (e.g. 10 meters/pixel)
- ModelTiepointTag: Ties a pixel coordinate to a real-world coordinate (e.g., "pixel 0,0 is at lon -118.5, lat 34.2")
- ModelTransformationTag: An alternative to the above two; a full 4x4 affine transformation matrix, used when the image is rotated.

These tags together define an [[Affine Transformation]], a 6-parameter mapping from pixel (col, row) -> real-world (X, Y):
- X = origin_x + col * pixel_width + row * row_rotation
- Y = origin_y + col * col_rotation + row * pixel_height

Bands:
- ==Each GeoTIFF has one or more bands; independent grids of values. Each band can have its own data type (uint8, float32, int16, etc.) and NoData value (a sentinel marking "no valid data here", like ocean pixels in a land-only DEM)==


## Internal Storage Layout:
- Classic GeoTIFFs store pixels in [[Scanline Order]] (row by row, left to right). This is efficient for sequential reads, but terrible for random spatial access (like service a web map tile from a small bounding box)
- [[Cloud-Optimized GeoTIFF]]s solve this problem by [[Tile]]ing internally and adding [[Overview]] pyramids.

![[Pasted image 20260423004231.png]]
