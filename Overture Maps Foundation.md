---
aliases:
  - Overture Maps
  - Overture
---
Site: https://overturemaps.org/

An industry consortium backed by Amazon, Microsoft, Meta, and TomTom launched in 2022 to produce a free, open, interoperable alternative to proprietary map data like [[Google Maps]] or [[HERE Technologies|HERE]], and to improve on [[OpenStreetMap]].

It produces a regularly updated dataset (released as [[GeoParquet]] files on [[Amazon S3|S3]]) covering the follow * Overture themes*:
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




> What we work on is bringing together open data and creating a referencable release that's interoperable. What we're reallyy doing there... I'm seeing the growth of data as being a challenge and being what keeps me up at night. It's not just about finding the best source for places, or the best source for road network, it's about being able to integrate any road network, sources... whetehr duplicative or not, to get to the point where you're producing a high-quality output in your release, and learning something by the agreement or disagreement of those sources and signals. The ingestion of these... and bringing them together... and understand the value of the quality that each one brings. What keeps me up at night is not just the technical challenge, but the organizational challenge of... these datasets, which have certain licenses and clear ownerships... and you're talking about what the best release is to put out htere, there are a lot of decisions to make, and many are use-case specific. How do we think of our data as being able to satisfy any number of usecases, rather than creating a very specific product to satisfy a specific use case."
> 
> ...
.
> A lot of what we do is iterative... and if something doesn't work, doesn't mean it won't work forever. If we build a component out to do validation of features, maybe we didn't have enough information to go on to build that out for a fully automated solution AT THAT TIME, but you can come back later... it's a constant iteration, constant evolution. Some of it's driven by technology, and other times by organizational dynamics. The flexibility to change in any given moment to reflect the state of the ecosystem, and not so much holding on to something that you've done and being unwilling to go a different route.
- Amy Rose, CTO @ Overture Maps