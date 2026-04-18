An open-source [[Data Integration]] platform that moves data from sources to destinations (databases, APIs, data warehouses, data lakes) through pre-built and custom connectors.
- It's the ==open source alternative to commercial [[ETL]]/[[ETL|ELT]] tools like [[Fivetran]] and [[Stitch]]==.
- The data integration problem (moving data from disparate sources into a central warehouse) is pervasive and generally hard at scale. Airbyte's connector ecosystem encapsulates complexity so that you don't have to rebuild it for every source.

Airbyte sits in the [[ETL|Extract, Transform, Load]] paradigm. It extracts data from a source, loads it into a destination, and ==leaves transformation to a downstream tool== (typically [[dbt]]).

It doesn't transform data in transit; it moves it faithfully and lets you decide what to do with it once it's in your warehouse.

```
  Source (Postgres, Salesforce, Stripe, S3, APIs...)
    → Airbyte (extract + load)                                           
      → Destination (BigQuery, Snowflake, Redshift, S3...)
        → dbt (transform)                                               
          → Analytics / ML
```

The ==heart of Airbyte is its connector catalog==:
- ==300+ pre-built connectors== for common sources and destinations, including [[PostgreSQL]], [[MySQL]], [[MongoDB]], Salesforce, HubSpot, Stripe, Google Analytics, GitHub, Slack, [[Amazon S3|S3]], Shopify, Zendesk, [[Google BigQuery|BigQuery]], [[Snowflake]], [[Amazon Redshift|Redshift]], [[Google Cloud Storage|GCS]], and hundreds more.
- ==Connectors are Docker containers, isolated, versioned, independently deployable. ==
	- Each container implements the ==Airbyte Protocol==, a standard interface defining how to discover a schema, read records, and handle state for incremental syncs.

# Sync Modes
- How data is replicated depends on the sync mode:
	- ==Full Refresh | Overwrite==: Fetch everything from the source every sync, replace the destination table. Simple but expensive for large tables.
	- ==Full Refresh | Append==: Fetch everything every sync, append to the destination. Builds a history of snapshots.
	- ==Incremental | Append==: Only fetch new/changed records since the last sync using a cursor field (e.g. updated_at). Much more efficient for larger tables.
	- ==Incremental | Deduped==: Incremental fetch, but deduplicate on the destination so each primary key has only one current row. ==The most common production mode.==


# Deployment
- Airbyte OSS (self-hosted, runs on [[Docker Compose]] or [[Kubernetes]])
- Airbyte Cloud (managed SaaS, Airbyte runs the infrastructure, cost per row synced)
- PyAirbyte (newer Python library for running connectors *directly in Python* without a running Airbyte server)

# Airbyte Protocol
- The open protocol that all connectors implement. Defines four operations:
```
  - spec: return the connector's configuration schema (what credentials/settings it needs)
  - check: validate that provided credentials work
  - discover: return the source's schema (what streams/tables exist, their fields and types)       
  - read: emit records as JSON lines, with state checkpoints for incremental syncs                 
```
Because the protocol is open, you can build custom connectors in any language; it just needs to be a Docker image that speaks the protocol.


# Relation to other tools:
- vs [[Fivetran]]: Fivetran is fully managed, more reliable, better for enterprise, and much more expensive.
- vs [[dbt]]: Complementary tech; Airbyte moves raw data into the warehouse, dbt transforms it. Commonly used together.











