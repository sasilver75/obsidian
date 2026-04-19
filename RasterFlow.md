References:
- [Introducing RasterFlow Blog Post](https://wherobots.com/blog/rasterflow-earth-observation-inference-engine/)
- [Wherobots Rasterflow: Planetary Scale Inference Engine for Earth Intelligence Video](https://www.youtube.com/watch?v=c3GWDtxVf1Es)


[[Wherobots]]'s serverless [[Remote Sensing|Earth Observation]] (EO) inference engine, automating the full pipeline from raw satellite/drone imagery to actionable geospatial insights at planetary scale.
- Core problem it solves: EO analysis traditionally requires rare specialist expertise and expensive custom infrastructure. RasterFlow makes it accessible to general data teams.

Core capabilities:
- Image prep: Ingesting satellite imagery, removes cloud cover/edge artifacts, builds inference-ready [[Mosaic]]s across time ranges.
- Distributed inference: Runs [[PyTorch]] models (custom or pre-built) across massive geographic areas (country-scale) in minutes/hours
- Built-in models: Includes [[Fields of the World]] (FTW) (crop boundary detection), Meta tree canopy height, road segmentation, and foundation model embeddings ([[OlmoEarth]], [[Clay]])
- Output: Vectorizes predictions into geometries and writes [[Apache Iceberg]] tables to [[Amazon S3|S3]].
	- Integrates with [[WherobotsDB]], [[Databricks]], [[Snowflake]], [[Google BigQuery|BigQuery]]


![[Pasted image 20260419113821.png]]

