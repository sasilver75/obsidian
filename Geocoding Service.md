A [[Geocode|Geocoding]] service converts between human-readable location descriptions (e.g. an address) and geographic coordinates, bridging between how humans describe place and how computers represent them spatially.

Two directions:
- [[Geocode|Forward Geocoding]]: Converting a location description to coordinates
```
  Forward geocoding — converts a location description to coordinates:
  "1600 Pennsylvania Avenue NW, Washington DC" → (38.8977, -77.0365)              
  "Eiffel Tower, Paris"                        → (48.8584, 2.2945)                
  "London, UK"                                 → (51.5074, -0.1278)
```
- [[Geocode|Reverse Geocoding]]: Converting coordinates to a human-readable description
```
  Reverse geocoding — converts coordinates to a human-readable description:       
  (40.7484, -73.9967) → "20 W 34th St, New York, NY 10001"                       
  (51.5007, -0.1246) → "Westminster Bridge, London, SE1 7PB"
```

Geocoding sounds difficult, but it's genuinely difficult at scale:
- Ambiguity
- Abbreviations and variations
- International address formats
- Data quality
- Informal addresses
- TRansliteration

# How it works, internally
- Essentially a ==text matching== + ==spatial lookup== pipeline..
	- ==Parsing==: Decompose input stirng into components (house number, street name, city, ...)
	- ==Normalization==: Standardize abbreviations, correct misspellings, expand acronyms
	- ==Candidate Retrieval==: Search a reference database of known addresses/places
	- ==Ranking and Disambiguation==: Score candidates by how well they match the query, apply geographic context
	- ==Interpolation==: For addresses that aren't in the database, interpolate position along a street segment (e.g. if you know where 100 Main St. is and where 200 Main Street at the end of the block, maybe 150 Main St. is right between them.)

Major Geocoding Services:
- Google Maps Geocoding API: Most widely used, highest coverage.
- Mapbox Geocoding API
- HERE Geocoding API
- ESRI World Geocoding Service
- Nominatim: Open-source, built on OpenStreetMap data. The go-to for open source geospatial projects.
- Pelias: Open-source, built on OpenSTreetMap data, designed for self-hosting... more performant than Nominatim at scale, more configurable.
- Photon: Lightweight open-source geocoder built on OSM data using [[ElasticSearch]]; fast, simple to deploy.
- What3Words: Divides the world into 3x3m squres, each assigned a unique 3 word combination ("filled.count.soap"). Designed for communicating precise locations in places without formal addresses. Controversial because proprietary, not open.