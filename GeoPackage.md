---
aliases:
  - .gpkg
---
A modern [[SQLite]]-based replacement for [[Shapefile]].
Unlike Shapefile, which is actually 3-7 files that must travel together, GeoPackage format files are ==single files==, support multiple layers, and handles projections propely.
- Increasingly common from government sources.
- Read natively in `geopandas`

> An open, standards-based, platform-independent, and portable geospatial data format designed for storing [[Vector]] features, [[Raster]] tiles, and raw data in a single, serverless SQLite database file. Developed by the [[Open Geospatial Consortium]] (OGC), it serves as an open-format alternative to proprietary formats like File Geodatabaes, offering superior flexibility, large storage capacity (up to 256TB), and widespread support across GIS software.
