---
aliases:
  - S3
  - Simple Storage Service
---
Amazon's [[Blob Storage|Object Storage]] service, one of the foundational infrastructure services of modern cloud computing.
It's the dominant cloud storage platform and the de-facto standard for storing large-scale data in the cloud.

Alternatives: [[Cloudflare R2]], [[Google Cloud Storage]] (GCS), [[Azure Blob Storage]], MinIO, Wasabi, Obstore

# Core Concepts
- ==Bucket==: A container for ==objects==. A globally unique name across all of AWS, belongs to a region. All objects in a bucket live in the same region (unless replicated). Buckets have policies, versioning settings, lifecycle rules, and access controls.
- ==Object==: The stored data, consisting of:
	- ==Key==: The full path string (e.g. `data/2023/06/sales.parquet`). This is the only identifier.
		- Looks like a file path, but it's just a string. S3 doesn't have real directories, though the S3 console GUI might make you think so.
	- ==Value==: The raw bytes of the object, up to 5TB/object
	- ==Metadata==: System metadata (content-type, size, last-modified, etag) + user-defined key-value paris.
	- ==Version ID== (if versioning is enabled)
- Offers 99.999999999% durability (11 nines); data is redundantly stored across multiple AZs within a region; losing data in S3 is extraordinarily rare.
- Offers 99.99% availability; the service will be accessible nearly all the time.
- Offers [[Strong Read-After-Write Consistency]] (previous to 2020 was only [[Eventual Consistency|Eventually Consistent]]). After a successful PUT, any subsequent GET returns the new object. After a DELETE, any subsequent GET returns 404.

# Storage Classes
- S3 Standard: Frequent access, highest cost per GB, lowest retrieval cost. Default.
- S3 Intelligent-Tiering: Automatically moves objects between tiers based on access patterns.
- S3 Standard-IA: Good for data accessed a few times per month.
- S3 One Zone-IA: Cheaper, but stored in one AZ, so less durable.
- S3 Glacier Instant Retrieval: Millisecond retrieval, good for data accessed once per quarter.
- S3 Glacier Flexible Retrieval: Retrieval takes minutes to hours, cold storage.
- S3 Glacier Deep Archive: Cheapest storage on AWS, retrieval takes up to 12 hours.

# Access Patterns
- S3 exposes a REST api with PUT/GET/HEAD/DELETE/LIST operations
- Range GET: Fetch a byte range of an object (see [[HTTP Range Request]])
- ==Multipart Upload==: For large objects (>100MB), split the object into parts, and upload in parallel.
- ==Presigned URLs==: Time-limited URLs that grant temporary access to a private object without requiring AWS credentially; Common if you want to allow a user to upload an item to a bucket from their browser, for instance.


# Object Storage vs [[File Storage]] vs [[Block Storage]]
- ==[[Block Storage]]== ([[Amazon Elastic Block Store|EBS]], local disk): Raw storage divided into fixed-size blocks. The OS manages a filesystem on top. Fast random reads/writes. Used for databases, OS volumes, anything needing low-latency random access.
- ==[[File Storage]]== ([[Amazon NFS|NFS]], [[Amazon Elastic File Store|EFS]]): Hierarchical filesystem with directories and files. Shared across multiple machines. Familiar interface, but scales poorly for large datasets.
- [[Blob Storage|Object Storage]] ([[Amazon S3|S3]]): Flat namespace of objects, each identified by a key (a string). No real directories, just keys that can contain slashes to *simulate* a hierarchy. No random writes; you write entire objects atomically. Infinitely scalable, cheap, highly durable. Designed for large files accessed as a hole, or via [[HTTP Range Request]]s.

```
Block:  read bytes 1024-2048 of a file (random access)
File:   open /data/myfile.csv, seek to line 500                                
Object: GET s3://bucket/myfile.csv (whole object or range)
```



# Geospatial Relevance
- S3 is the storage layer for nearly all [[Cloud-Native Geospatial]] work:
	- [[Cloud-Optimized GeoTIFF]] (COG) files are served directly from S3 via range requests.
	- [[Zarr]] stores with millions of chunk objects.
	- [[SpatioTemporal Asset Catalog|STAC]] catalogs as static JSON on S3.
	- [[PMTiles]] archivse are served from S3 via range requests
	- [[GeoParquet]] files for analytical workloads via [[DuckDB]]
	- [[Entwine Point Tiles]] (EPT) stores
	- [[Sentinel|Sentinel-2]], [[Landsat]], and other open satellite data on [[Amazon Open Data]]