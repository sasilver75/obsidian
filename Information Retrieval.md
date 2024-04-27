


## Information Retrieval Metrics
There are many ways to assess the quality of IR systems
	1. ==Accuracy-style metrics==: these will be our focus
	2. ==Latency==: Time to execute a single query; incredibly important in industrial contexts -- users demand low-latency systems. You often see accuracy/latency paired.
	3. ==Throughput==: Total queries served in a fixed time, perhaps via batch processing. Sometimes related to latency; you might sacrifice per-query speed to process batches efficiently.
	4. ==FLOPs==: A hardware-agnostic measure of compute resources; Hard to measure/reason-about, but might be a good summary method
	5. ==Disk usage==: For the model, index, etc. If we're going to index the whole web, the cost of storing it all on disk might be very large.
	6. ==Memory usage==: For the model. index, etc.
	7. ==Cost==: Total cost of deployment for a system; summarizes all of 2-6, in a way.
		- If we want great latency, so we store everything in memory, but this is very expensive.
		- We might then cut costs my making our system smaller, but that would lead to a loss in accuracy.

Specific metrics:
- [[Reciprocal Rank]] (RR)
- [[Success]]
- [[Precision]], [[Recall]], [[F1 Score]], [[Average Precision]]

## So which metric should we use?
1. Is the cost of scrolling through K passages low? Then perhaps Success@K is fine-grained enough.
2. Are there multiple relevant documents per query? If so, Success@K and RR@K may be too coarse-grained...
3. Is it more important to find *every relevant document?* If so, favor [[Recall]] (Cost of missing documents is high, cost of review is low)
4. Is it more important to *only view relevant documents?* If so, favor [[Precision]] (Cost of review is low)
5. [[F1 Score]] F1@K is the harmonic mean of Prec@K and Recall@K. It can be used when there are multiple relevant documents but their relative order above K doesn't matter much.
6. [[Average Precision]] will give the finest-grained distinctions of all metrics discussed; it's sensitive to rank, precision, and recall.
