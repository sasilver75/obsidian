---
aliases:
  - Overture Maps
---
Site: https://overturemaps.org/

An industry consortium backed by Amazon, Microsoft, Meta, and TomTom launched in 2022 to produce a free, open, interoperable alternative to proprietary map data like [[Google Maps]] or [[HERE Technologies|HERE]], and to improve on [[OpenStreetMap]].

It produces a regularly updated dataset (released as [[GeoParquet]] files on [[Amazon S3|S3]]) covering:
- Places (POIs like restaurants, shops, landmarks)
- Buildings (Global building footprints, incorporated [[Microsoft Building Footprints]] and [[OpenStreetMap]])
- Transportation (road networks, connectors, turn restrictions)
- Administrative boundaries (countries, states, localities)
- Base (land use, land cover, water features)

Has a CDLA Permissive 2.0 license.

Context:
- ==[[OpenStreetMap]] (OSM) is the incumbent open map, but its schema is inconsistent and its data quality varies wildly be region==. 
	- ==Overtures aims to be more consistent==, reliable, schema-normalized (particularly for the building and places layers where OSM has gaps)

Data comes from:
- [[OpenStreetMap|OSM]] itself: Overture ingests and normalizes OSM data as a major input source.
- Microsoft's [[Microsoft Building Footprints]], which is derived from satellite imagery
- Meta contributes their own road and places data (derived from their mapping efforts and AI)
- TomTom contributes commercial map data from their existing datasets

So in a sense, ==Overture is less a new mapping project and more a data normalization and reconciliation layer on top of existing sources==, with a consistent schema and open license.