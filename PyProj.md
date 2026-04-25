
A Python wrapper around the [[PROJ]] library, the ==standard tool for coordinate reference system definitions and transformations==.
- In short: Given a point in CRS A, give me its coordinates in CRS B.

[[Geopandas]] uses it under the hood whenever you call `.to_crs(...)`

```python
from pyproj import Transformer

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32611")
x, y = transformer.transform(34.05, -118.25)  # lat/lon → UTM meters
```



