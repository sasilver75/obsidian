A probabilistic data structure, designed for ==frequency estimation== in data streams! 
It's particularly well-suite for ==situations where you need to keep track of the frequencies of various elements== (such as *words in a document*, or *IP addresses accessing a website*).

# How it works

Consists of a two-dimensional array of counters, along with several hash functions (e.g. a few hundred), with each hash function corresponding to one row of the array.

==Write operation==: When a new element is added, it's hashed by each of the hash functions. The resulting hash values are used as indices to increment counters in each row of the array.

==Read operation==: To estimate the frequency of an element, the element is hashed with the same hash functions; the *MINIMUM* value among the corresponding counters is taken as the estimated frequency of the element.
- Note that the CountMinSketch can only ==*overestimate frequencies*==, not underestimate them; this is due to the nature of using the minimum counter values as estimates; these can be influenced by hash collisions.

![[Pasted image 20240422142005.png]]


----
Aside: [[HyperLogLog]] vs Count-Min Sketch:
> For example, Put 3 of the item 'A' (where each 'A' is indistinguishable from another 'A') and similarly 5 of item 'B' into a Count-Min Sketch and a HyperLogLog. When you query the Count-Min Sketch for 'A', you can probably get back 3. You can also query for 'B' and get back 5. With you query the HyperLogLog you query how many unique/indistinguishable items are in the set, in this case 2. The HyperLogLog would return a number near 2.
----
