[Github Link](https://github.com/corteva/rioxarray)

==[[Xarray]] is great== for N-dimensional labeled arrays (especially time series of raster data), ==but it has no native concept of [[Coordinate Reference System|CRS]], Geotransforms== (affine transformations mapping pixel coordinates to real world coordinates), ==or Spatial Operations== (intersects, contains, etc.). 
- `rioxarray` is a ==Python library bridges that gap== by adding a `.rio` accessor to xarray DataArray and Dataset objects, bringing [[rasterio]]/[[Geospatial Data Abstraction Library|GDAL]]-backed geospatial capabilities into the Xarray world!
