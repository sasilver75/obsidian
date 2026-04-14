---
aliases:
  - Ray Data
  - Ray Train
  - Ray Serve
---


"Distributed Python, generalized"
- Not just for data analysis, but for any Python workload.
- ==Take any arbitrary Python functions and run them in parallel across machines.==
	- Where [[Apache Spark|Spark]] and [[Dask]] are focused on data transformations, Ray is focused on ==tasks== and ==actors==.

How it works:
- Decorate functions with `@ray.remote` to make them run as distributed tasks.
- Call them with `.remote()` instead of normally -- returns a `future` (a result not yet computed).
- Ray handles scheduling, memory sharing between workers, and fault tolerance.
- Has higher-level libraries built on top:
	- `Ray Data`: Like Dask Dataframes
	- `Ray Train`: Distributed ML training
	- `Ray Serve` Model Service

Strenghts:
- General-purpose: Works for ML training, simulations, inference
- Very Pythonic: Feels natural, not like a weird data pipeline DSL
- Strong ML ecosystem (integrates with [[PyTorch]], [[XGBoost]], etc.)
- Good at heterogenous workloads (mix of CPU/GPU tasks)

Weaknesses:
- Les mature for pure SQL/analytics workloads vs [[Apache Spark|Spark]]
- Smaller community than other tools lke Spark


![[Pasted image 20260414121252.png]]
When you'd use each:
- Spark:
	- You work at a company with a data warehouse, Databricks or EMR clusters, and
     petabyte-scale logs. Standard choice for production ETL at scale.
- Dask:
	- You're a data scientist with a 50GB CSV that won't fit in pandas. You want to
     run your existing pandas code without rewriting it.
- Ray:
	- You're training ML models and want to parallelize hyperparameter search across
     GPUs. Or you have a general Python workload (simulations, web scraping at scale)
    that isn't really "data transformation."


