
Space-efficient computing tools that sacrifice perfect accuracy to provide fast, approximate answers to large-scale data queries, typically using randomization and hashing so that they use a fraction of the memory that traditional, exact data structures (eg [[Hash Set]]s) require.

Used in situations where memory is a bottleneck, or when you want to save on latency via avoiding expensive operations by first checking a probabilistic data structure, doing operations like:
- Checking for set membership
- Counting unique elements (set cardinality)
- Determining element frequency


Examples:
- [[Bloom Filter]]: "Is `x` absent or maybe present?"
- [[Cuckoo Filter]]: "Is `x` absent or maybe present, with deletion support"
- [[Count-Min Sketch]]: "About how many times did `x` occur?"
- [[HyperLogLog]]: "About how many unique things occurred?"
- [[Locality-Sensitive Hashing]]: "Which items are probably similar to `x`?"
- [[MinHash]]: "How much do these two sets overlap?"
- [[SimHash]]: "Are these feature vectors/texts near-duplicates?"





