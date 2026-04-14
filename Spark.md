Built at Berkeley in 2009 and now the industry standard for large-scale data processing.

Idea:
- Distribute a dataset across a cluster of machines and run transformations on it in parallel. 
	- You write code that looks like it operates on one big dataset, and Spark figures out how to split the work across nodes.

How it works:
- Data is represented as a ==DataFrame== (or RDD, the lower-level abstraction)
- Transformations are ==lazy==: For `df.filter(...).groupBy(...).count()`, nothing runs yet; Spark just builds a query plan.
- When you call an ==action== 