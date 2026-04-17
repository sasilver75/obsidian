---
aliases:
  - HDF5
---
A binary file format and library for storing large, complex scientific data.
One of the foundational formats in scientific computing, predating [[Network Common Data Form|NetCDF]]-4, which is literally built on top of it.

It's essentially a ==filesystem inside a file:== it has a hierarchical group structure (like directories), datasets (like files), and attributes (like metadata tags) all packed into a single binary file.
```
  /
  ├── group: climate/
  │     ├── dataset: temperature   [time × lat × lon, float32]
  │     ├── dataset: humidity      [time × lat × lon, float32]
  │     └── attrs: {"source": "ERA5", "year": 2023}
  ├── group: ocean/
  │     ├── dataset: SST           [time × lat × lon, float32]
  │     └── group: currents/
  │           ├── dataset: u_velocity
  │           └── dataset: v_velocity
  └── attrs: {"institution": "ECMWF"}
```

This flexibility (multiple datasets, arbitrary nesting, mixed data types) in one file is HDF5 defining characteristic, and why it's ==used for data that doesn't fit neatly into one array==.

Other features:
- Chunking and compression
- Parallel I/O
- Arbitrary data types
- Single writer/multiple reader

==Designed for local POSIX filesystems and HPC shared storage==, but on [[Blob Storage|Object Storage]], it has the ==same fundamental issues/problems as [[Network Common Data Form|NetCDF]]==:
-   Single large file requires many small range requests to navigate the internal
  B-tree structure
- The HDF5 library makes many small reads to traverse metadata before getting to data
- Not designed for parallel object storage semantics


Again, [[Zarr]] is clean solution: Same logical model, designed from scratch for object storage.
- The community is slowly migrating, but the HDF5/NetCDF archive is enormous, and won't be reformatted overnight.
