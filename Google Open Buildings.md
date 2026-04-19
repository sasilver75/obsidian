A global open dataset of building footprints extracted from satellite imagery using ML
- This is Google's equivalent of [[Microsoft Building Footprints]], with particular strength in the global south. *"Like Microsoft Building Footprints but trained specifically to find the informal settlements and rural compounds that other models miss, and at nearly twice the count.*

Released by Google Research in 2021, initially focused on Africa.
- Expanded to cover South/Southeast Asia, Latin America, and now approaching global coverage.
- Motivated by the fact that these regions are severely undermapped in [[OpenStreetMap]] and other sources.

Coverage:
- ~1.8B buildings globally
- Significantly larger count than Microsoft's dataset, partly because it covers denser informal settlements better.

Released as [[Comma-Separated Values|CSV]] with [[Well-Known Text|WKT]] geometries (awkward), but also available as [[GeoParquet]] via [[Google Cloud Storage|GCS]] with a [[SpatioTemporal Asset Catalog|STAC]] catalog available.


