A global open dataset of building outlines extracted from satellite imagery using deep learning, one of the largest open building datasets in existence.
- Google has an alternative to this in [[Google Open Buildings]], which is trained on different imagery and has particularly strong coverage in Africa, South/Southeast Asia, and releases height estimates. Complementary to this dataset.

Released as [[GeoJSON]] + [[GeoParquet]], free to use.
- Released as GeoJSON per state/region, but also available as GeoParquet on [[Microsoft Planetary Computer]], making it cloud-native queryable via [[SpatioTemporal Asset Catalog|STAC]].

Coverage:
- US: ~130M buildings over entire country
- Canada: ~12M buildings
- Uganda and Tanzania: Earlier Africa release
- Global: Ongoing expansion; they've released continent-scale datasets


Each feature is just a polygon; building footprint outline.
- Attributes:
	- Geometry (the polygon)
	- Capture date
	- Confidence score (some releases)

Use cases:
- Population estimation (building count * occupancy estimate)
- Disaster response (pre-event baseline for damage assessment)
- Uber growth tracking over time
- Filling OSM data in undermapped regions
- Training data for other ML models






