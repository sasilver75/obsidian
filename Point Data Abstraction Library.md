---
aliases:
  - PDAL
---
Basically ==the [[Geospatial Data Abstraction Library|GDAL]] of point clouds==; Open source C++ library with a Python interface that ==reads/writes virtually every point cloud format== ([[LASer|LAS]], [[LAZ]], [[Cloud-Optimized Point Cloud|COPC]], [[Entwine Point Tiles|EPT]], PLY, PLTS, etc.) and provides a pipeline-based processing framework.

You define a JSON pipeline of readers, filters, and writers:
```
{
    "pipeline": [
      {"type": "readers.las", "filename": "input.laz"},
      {"type": "filters.ground"},
      {"type": "filters.range", "limits": "Classification[2:2]"},
      {"type": "writers.gdal", "filename": "dtm.tif", "resolution": 1.0}
    ]
}
```

==PDAL is the foundation most open-source LiDAR tools are built on or interoperate with.==

