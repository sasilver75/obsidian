---
aliases:
  - Spacial Reference Identifier
  - Spatial Reference ID
---
It's a database-internal integer used to identify a [[Coordinate Reference System|Coordinate Reference System]] in a spatial database like [[PostGIS]].

- A [[Coordinate Reference System|CRS]] is the abstract concept (full mathematical definition of how coordinates map to locations on Earth)
- [[EPSG Code|EPSG]] is the organization (now part of IOGP) that maintains a registry of numbered CRS definitions (e.g. 4326, 3857).
- [[SRID]] is what a database uses internally to refer to one of those definitions.

In PostGIS, the `spatial_ref_sys` table ships pre-loaded with thousands of EPSG definition, and the `srid` column in that table IS the EPSG code -- ==PostGIS simply adopted EPSG numbers directly as its SRIDs, so in practice, SRID and EPSG Codes are the same number for any standard CRS!==

Note though that SRID is a generic databse concept, and other spatial databases like Oracle Spatial use their own SRID numbering schemes that don't match EPSG codes for coordinate reference systems.

