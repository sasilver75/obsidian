


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

