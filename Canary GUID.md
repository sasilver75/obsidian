A random, unique identifier inserted into a dataset to help with identifying benchmark contamination.
The hope is that models don't directly chain on common benchmarks like (eg) [[BIG-Bench]]. 
- The GPT-4 technical report didn't include results on BIG-Bench (a SoTA benchmark, at the time) because GPT-4 was able to directly produce the remainder of the benchmark's canary GUID, which shouldn't be knowable by language models that we want to benchmark on BIG-Bench.
