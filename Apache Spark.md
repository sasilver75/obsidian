---
aliases:
  - Spark
  - PySpark
  - SparkSQL
---

Built at Berkeley in 2009 and now the industry standard for large-scale data processing.
Natively in Scala, but has Python (PySpark), Java, and R APIs. 

Idea:
- Distribute a dataset across a cluster of machines and run transformations on it in parallel. 
	- You write code that looks like it operates on one big dataset, and Spark figures out how to split the work across nodes.

How it works:
- Data is represented as a ==DataFrame== (or RDD, the lower-level abstraction)
- Transformations are ==lazy==: For `df.filter(...).groupBy(...).count()`, nothing runs yet; Spark just builds a query plan.
- When you call an ==action== like `.show()` or `.write()`, THEN Spark executes the whole plan at once, after optimizing it.
- The ==driver== (your program) coordinates, while ==executors== (workers on other machines) do the actual work.

Strengths:
- Battle-tested at massive petabyte scale
- Rich ecosystem: SparkSQL, MLlib, StructureD Streaming
- Deep optimization engine (Catalyst optimizer)

Weaknesses:
- Heavy, JVM-based; slow to start, memory overhead
- Painful to run locally for development
- Overkill for anything under ~100GB




