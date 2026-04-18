---
tags:
  - Catalog
---


HUDI = "Hadoop Upserts Deletes and Incrementals"
- An open-source table format and data management framework, originally built by Uber in 2016 and open-sourced in 2019.

It's the third leg of the "**==table format wars==**" alongside [[Delta Lake]] and [[Apache Iceberg]]

Uber built Hudi to solve a very specific problem: They needed to ingest billions of trip events per day into their data lake with low latency, while also supporting record-level updates and deletes.
- Hudi was designed for high-frequency upserts from streaming sources, which distinguishes from Delta and Iceberg, which were more focused on batch analytics with ACID guarantees on top.


Hudi's most distinctive feature — two fundamentally different storage layouts optimized for different workloads:                                                                          
-   ==**Copy-on-Write (CoW)**==                                                                               
  - All data stored as Parquet files                                                            
  - On write: read affected files, merge updates, write new Parquet files                           
  - Reads are fast (pure Parquet, no merging needed)                                            
  - Writes are expensive (must rewrite files for every update)                                      
  - Good for: batch ingestion, read-heavy analytics                                                 
-   ==**Merge-on-Read (MoR)**==                                                                               
  - Base files are Parquet (for bulk data)                                                          
  - Updates written to **delta log files** (Avro format, append-only)                                   
  - Reads merge base + delta logs on the fly                                                    
  - Writes are very fast (just append to delta log)                                                 
  - Reads are slower (merging overhead)                                                             
  - Periodically **compacted** — delta logs merged back into Parquet base files                    
  - Good for: streaming ingestion, frequent updates, near-real-time


```
  Hudi is the right choice when:

  - You have high-frequency record-level upserts from streaming sources (Kafka, Flink)              
  - You need near-real-time data availability (MoR with frequent compaction)                        
  - You need efficient CDC — downstream systems consuming only changed records                      
  - You're already in a Spark-heavy environment                                                     

  Hudi is less compelling when:                                         
  - You have primarily batch workloads with occasional updates          
  - You need broad multi-engine support (Iceberg wins here)              
  - You're on Databricks (Delta wins there)                              
  - Your team doesn't have the operational experience to tune compaction and clustering
```


In practice, Hudi is most common at companies with large-scale streaming data pipelines — ride-sharing, e-commerce, ad-tech — where the upsert performance and incremental pull capabilities justify the operational complexity. For most analytics-first data teams, Iceberg or Delta is simpler and sufficient.







