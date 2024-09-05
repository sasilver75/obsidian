A probabilistic algorithm used for estimating the number of distinct elements (cardinality) in a large dataset or stream of data (unique visitors to websites, unique items in a caching scenario, user counts in streaming data, IP addresses for network traffic monitoring).
- Note that "HyperLogLog++" is an improved version with better accuracy for small cardinalities.

Sacrifices exact counting for extreme efficiency, thus isn't appropriate when *exact counts* are required.

Process:
- Hashes each element to a binary string.
- We focus on the position of the leftmost "1" bit in the binary hash, with the intuition that the probability of seeing a hash with n leading zeroes is 2^-n
- The binary hash is split into two parts:
	- The first few bits (typically 14) are used as a bucket identifier (2^14 buckets). These buckets are used to improve overall estimation accuracy.
	- The remaining bits are used for the actual estimation.
- Bucket updates:
	- For each element, its bucket is determined by the first part of the hash.
	- We count the number of leading zeroes in the second part of the hash.
	- We compare this count to the value currently stored in the corresponding bucket, and keep the larger of the two counts.
- Estimation:
	- After processing all elements, each bucket contains the maximum observed leading zeros.

So HyperLogLog estimates the total number of distinct elements across all inputs (eg unique site visitors), but can't tell you many times (eg) user A, B, or C *specifically* visited.

Q: If we're just trying to count the number of unique items in a stream of data, why don't we use something like a hashset?
A: Hashsets (or Hashtables) have O(n) memory usage for n unique elements, and provide an exact count. HyperLogLog instead uses a fixed amount of memory, regardless of input size: O(1), typically a few kilobytes. As a downside, it provides an estimate with small error margins.


----
Aside: HyperLogLog vs [[Count-Min Sketch]]:
> For example, Put 3 of the item 'A' (where each 'A' is indistinguishable from another 'A') and similarly 5 of item 'B' into a Count-Min Sketch and a HyperLogLog. 
> When you query the Count-Min Sketch for 'A', you can probably get back 3. You can also query for 'B' and get back 5. 
> When you query the HyperLogLog you query how many unique/indistinguishable items are in the set, in this case 2. The HyperLogLog would return a number near 2.
----
