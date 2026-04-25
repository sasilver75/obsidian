---
aliases:
  - .gpkg
---
A file format for [[Vector]] data, and a modern [[SQLite]]-based replacement for [[Shapefile]].
- Not cloud-optimized, since it's stored internally as a SQLite database, the entire file must be downloaded to read any part of the file, and it requires a server (so the frontend can't just [[HTTP Range Request]] from [[Blob Storage|Object Storage]]).
Unlike Shapefile, which is actually 3-7 files that must travel together, GeoPackage format files are ==single files==, support multiple layers, and handles projections properly.
- Increasingly common from government sources.
- Read natively in `geopandas`

> An open, standards-based, platform-independent, and portable geospatial data format designed for storing [[Vector]] features, [[Raster]] tiles, and raw data in a single, serverless SQLite database file. Developed by the [[Open Geospatial Consortium]] (OGC), it serves as an open-format alternative to proprietary formats like File Geodatabaes, offering superior flexibility, large storage capacity (up to 256TB), and widespread support across GIS software.



![[Pasted image 20260424223846.png]]
