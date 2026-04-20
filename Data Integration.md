Moving data from where it lives to where it you need it.

There are some sub-categories:
- [[ETL]]: Pull data from source, transform it *before* loading into the destination. Transformations happen in the pipeline itself.
	- Often a custom script or tool like Informatica
- [[ETL|ELT]]: Pull data raw, dump it directly into the [[Data Warehouse]], *THEN* transform it using SQL. Works because cloud warehouses are cheap and powerful enough to do the transform step themselves.
	- Tools like [[Fivetran]], [[Airbyte]] don't transform, they just move raw data. Sort of the L in ELT.
	- Tools like [[dbt]] are typically used to do the transformations.
- [[Change Data Capture]] (CDC): Taps into databases transaction logs (same one they use internally for crash recovery; every write to the DB appends to this log), reads it, and re-publishes those events in real-time. Feeds into "Stream ingestion"
	- Operationally complex; Uses tools like [[Debezium]], which output to a distributed log like [[Kafka]], which then get connected to other data systems (e.g. CDCing Postgres data into ElasticSearch for a search functionality).
- Stream Ingestion: Where real-time data lands before being processed.
	- [[Kafka]] and [[Amazon Kinesis|Kinesis]] are message queues/event buses where producers write events and  consumers read them independently at their own pace.
	- Again, CDC feeds into here. Something like Debezium publishes DB changes onto a Kafka topic, then downstream consumers (e.g. [[Apache Flink|Flink]]) read from it.
	- [[ETL|Reverse ETL]]: The data warehouse was originally a read-only analytics destination; Reverse ETL flips this, taking data out of the warehouse and pushing it back into operational tools.
		- Tools like Census and Hightouch do this.
		- It became a distinct category because the warehouse became the "source of truth" and teams wanted operational systems to reflect that.
		- Examples
			- Sync customer segments from Snowflake -> Salesforce for sales reps
			- Push churn scores from BigQuery -> Intercom to trigger emails
			- Load computed metrics -> HubSpot for marketing.

