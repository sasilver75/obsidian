---
aliases:
  - GERS
---
An open ==framework introduced by the [[Overture Maps Foundation]] for assigning stale, globally-unique identifiers to real-world geographic features== (buildings, roads, places, administrative boundaries) so that the same physical entity can be consistently referenced and linked across different datasets from different sources.
- An attempt to provide a ==common identity layer,== a shared reference system that different datasets can use to say "this is the thing!"
- The deeper intent is for GERS to become a ==join key for the global geospatial ecosystem== without fuzzy matching, geometry inspection, or manual reconciliation.

Problem: The same building, road, or place exists in many different datasets simultaneously ([[OpenStreetMap]], government data, commercial POI databases, satellite-derived footprints, Yelp, [[Google Places]])... each with their own internal ID that means nothing to the other datasets:

```
  The Empire State Building:
    OpenStreetMap:        way/34633854                                          
    Google Places:        ChIJaXQRs6lZwokRY6EFpJnhNNE                            
    Foursquare:           40a55d80f964a52020f31ee3                               
    NYC Building database: BIN 1015862                                           
    Overture Maps:        [GERS ID]
```

# How it Works
- Overture assigns a GERS ID to every feature it publishes
- These features are:
	- Globally unique
	- Stable across releases
	- Persistent through edits (e.g. if the feature' geometry or attributes change)
	- Encoded with provenance (where did hte feature come from, how was it derived)

Structure:
- 128-bit identifiers encoded as strings, encoding:
	- A theme prefix (which Overture theme the feature belongs to)
	- A unique version identifier component
	- Version/provenance information
Example: `08f2a100d2c6c6b5040792b1c95649d9`
They're designed to be opaque; you shouldn't parse meaning from the string itself.


Overture provides an open source `overture-conflation` Python toolkit that takes your existing dataset and Overture data and fines matches, outputting a mapping table of your IDs to GERS IDs.







