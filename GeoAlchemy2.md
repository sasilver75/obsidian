An extension for [[SQLAlchemy]] that adds [[PostGIS]] geometry type support.
- Without it, SQLAlchemy doesn't have any idea what a `GEOMETRY(Point, 4326)` column is.

```python
from geoalchemy2 import Geometry
from sqlalchemy.orm import Mapped, mapped_column
from typing import Any, Optional

class SR311(Base):
    __tablename__ = "sr_311"

    # spatial_index=False because we define the GiST index explicitly below.
    # By default GeoAlchemy2 auto-creates one — if you also define it manually
    # you get a "relation already exists" error during migration.
    geom: Mapped[Optional[Any]] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=False),
        nullable=True
    )
```
Above:
- Using SQLAlchemy and GeoAlchemy2 to declare a Geometry-type column on a Model.
	- The `Geometry` type takes a geometry type string (POINT, POLYGON, MULTIPOLYGON) and an [[SRID]] (an identifier for a [[Coordinate Reference System|Coordinate Reference System]]; 4326 is [[WGS84]], the standard geodetic datum used for global positioning, navigation, and mapping.)


### The 'spatial_index=False' Gotcha
- GeoAlchemy2 defaults to `spatial_index=True`, which makes it auto-create a [[Generalized Search Tree|GiST]] index named `idx_{tablename}_{columnname}` when the table is craeted.
	- If you also manually create an `Index(...)` in the `__table_args__` class attribute of your model, you get a duplicate index error at migration time.
	- So if you're going to define your GiST index explicitly, always set `spatial_index=False` on your actual column.



Note that GeAlchemy stores geometries as [[Well-Known Binary|WKB]] [[Hexadecimal]] strings internally. You rarely touch these directly.

```python
# Writing — pass a WKB element or use ST_MakePoint in raw SQL
from geoalchemy2.shape import from_shape
from shapely.geometry import Point

geom_value = from_shape(Point(-118.25, 34.05), srid=4326)
row.geom = geom_value

# Reading — convert back to a Shapely object
from geoalchemy2.shape import to_shape
point = to_shape(row.geom)
print(point.x, point.y)  # -118.25, 34.05
```


### GeoAlchemy2 / Alembic Autogenerate
- Note that Alembic's revision autogeneration feature doesn't always understand GeoAlchemy2 types -- it might generate the column correct but forget to add `import geoalchemy2` to the migration file, so you may have to do this manually if the generated migration touches geometry columns.


