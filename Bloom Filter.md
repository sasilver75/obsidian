
Bloom Filters are ==probabilistic data structure== used to ==test whether an element is a member of a set==.
They're highly-efficient in terms of space, especially when dealing with large datasets.

The *trade-off* for this efficiency is a *==certain probability of false positives==* (sometimes it will say that an element *is* in a set, even though it's really not, in actuality).
However, it will *==never produce false negatives==* (meaning *if an item is actually in the set, the bloom filter will always say it in the set.*)

The main advantage of a bloom filter lies in its space-efficiency and speed, for membership tests in large sets.
It's used in various scenarios:
- Web caching systems to avoid storing duplicate web pages.
- Spell checkers, to quickly check whether a word is in a dictionary.
- Database systems, for quick lookups in-memory that avoid expensive disk or network operations that would occur if the item is definitely *not* in the set.

# How it works
A bloom filter uses ==multiple hash functions== to ==map each element in the set to *several positions* in a bit array== of $m$ bits.

Initially, all bits in the array are set to 0.
- ==Write operation==: When an element is added to the set, the bits at the positions indicated by the hash functions are set to 1.
- ==Read operation==: To check if an item is in the set, the item is hashed with the same hash functions, and the bits at the resulting positions are checked.

The possibility of false positives arises because different elements could result in the same bits being set to 1 by the hash functions (hash collision).
The probability of false positives can be reduced by ***increasing the size of the bit array*** and ***increasing the number of the hash functions***.

Bloom filters ==do not support removal==; it would be impossible to remove an element that's been added without introducing false negatives, because flipping a bit from 1 -> 0 could affect other elements that hash to that bit.


![[Pasted image 20240422142046.png]]

# Bloom Filter extensions
- Counting Bloom Filters (allows for removal of elements, but requires more space)
- Scalable Bloom Filters (dynamically adjust size as elements are added, maintaining a constant false-positive probability)
- ==Cuckoo Filter== (allows for removal of elements, has better space efficiency, and provides similar/better performance.)
- Bloomier Filters (Not only tells when an item is in the set, but allows storing some additional metadata with the item)
- ==Spectral Bloom Filters== (Designed to not only report if an item is in a set, but *how many times it might have been added*; good for occurrances that require count of occurrances as well as membership) -- this is similar in some senses to the functionality of [[Count-Min Sketch]] datastructures.
- Partitioned Bloom Filters (Reduces the chance of hash collisions, improving the accuracy of the filter at the cost of slightly increased complexity)
- Dynamic Bloom Filters (Similar to scalable bloom filters)

