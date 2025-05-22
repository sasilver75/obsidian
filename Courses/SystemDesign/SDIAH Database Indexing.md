SDIAH Link: https://www.hellointerview.com/learn/system-design/deep-dives/db-indexing

1. What is an Index?
2. What problem do they solve?
3. When should you use Indexes?

In a database, data is arranged in [[Page]]s, which are usually 8KB of data.

When you want to find a particular item in your database **when you aren't using indexing**, what happens is:
![[Pasted image 20250520171732.png]]
- We pull a page into RAM, scan through all ~100 rows or items, and if we don't find it, we put that page back, pull in the next page, scan it, etc. until we eventually find it.
- If we had **100M** users in our database and each page was **100 rows**, then that's **1M** pages. If each round-trip from SSD to RAM is is ~100ms, then that's **100 seconds** in the worst case for us to find the item we're looking for! (This sort of ignores some other database optimizations, but the point stands that this is too long!)

# So what are Indexes?
- [[Index]]es are data structures that are stored on disks and act as a map to tell us on which page some item exists in the database.
- When a new query comes in for a particular item, we:
	- Pull our index into RAM
	- Check the index to see which page that item lives on
	- Pull only that particular page into RAM
- So instead of reading 1M pages, instead we use the index to tell us exactly which page to look at.

# B-Trees
[[B-Tree]]s are the most popular index; let's look at this!
![[Pasted image 20250520172120.png]]
- Each node in the the is sorted list of values, with pointers to another page on disk
	- Either a child node (with more pointers), or an actual data page (e.g. at the bottom in the picture)
- So above, we have a User table, and we've built an index on Age.
	- If we run the query above, the first thing we do is pull the root node into memory (Note that each of these nodes is a [[Page]] on disk!).
	- We look at this node and say that we want 51, which is between 50 and 90, so we pull in the relevant node/page into memory!
	- We then look at that node and say "51 is less than 55, so this corresponds to page 3!"
	- So we pull Page 3 into memory, and this is where all of the users that are Age 51 exist!

What if we had the following query?
![[Pasted image 20250520172350.png]]
See here that we're looking for all users with ages GREATER THAN 51!
- In this case, we pull in our root node, then pull in BOTH of these descendant blocks, since they're both greater than 50.
- Then we follow all 7 pointers from these descendant nodes, and pull those 7 pages into memory, since these pages store all of the users whose age is greater than 51.


# Hash Indexes
- A [[Hash Index]] is really straightforward; it's just a HashMap that exists on disk!
- ![[Pasted image 20250520172539.png]]
- If we're looking for a given email, we just pass that user's email into a hash function, and we have a hashmap that matches this hash key to a value of where that page exists on disk!
- In reality, Hash Indexes are ==rarely used in production databases!==
	- They offer O(1) lookups, but ==B-Trees perform nearly as well for exact matches, but also support range queries and sorting as we saw above!==
	- So the ==only place where you'll often see these Hash Indexes are for **in-memory stores** like [[Redis]], where these disk access patterns aren't as important.==


# Geospatial Indexes
- There are some places where we wouldn't want to use a B-Tree, like anytime where we have geospatial data!
![[Pasted image 20250520172737.png]]
- In a Yelp SD interview, we might want everyone within a certain radius!
	- If we had a query like the one above, ==B-Trees excel at one-dimensional data, but not two-dimensional data like this!==
	- This query would give us ALL OF THE RED DATA FOR LATITUDE and ALL OF THE BLUE DATA FOR LONGITUDE! 
		- We'd then load both of these into memory, and do a fairly expensive merge before returning the data that we want.
- There are some algorithms that are designed to be good for these, related to [[Geospatial Index]]es:
	- [[Geohash]]ing, [[QuadTree]], and [[R-Tree]]!


## GeoHashing
![[Pasted image 20250520173041.png]]
- In GeoHashing, we take a map of the world, and split it into four parts
- We then recursively split each of these cells, doing the same thing, such that the 2 cell is broken into 20, 21, 22, 23. 
- By continuing to do this, we get increasing levels of precision!
	- So New Mexico might be ~31, but Albequerque itself might be ~310!
- **==Now all nearby locations likely share a common prefix!==**
- If we want to find everything near Chichuahua (330), we just check the prefixes that are near it (321, 331, 332, 312).
- So we create the geohashes, and then build a [[B-Tree]] on these geohashes, allowing us to do these range queries and point lookups on geohashes!
- **==NOTE:==** We used these simple numbers (312) for this example, but what we actually do in reality is [[Base32]] encode them, so Los Angeles is `9qh16`.


## Quad Trees
- [[QuadTree]] are similar to GeoHashing, in that we split the world recursively, but there are a couple subtle differences; we map this recursive splitting to a tree, not to a one-dimensional string.
- We only need to go deeper in this tree in the areas where we have high density!
![[Pasted image 20250520173454.png]]
- Say each dot is a business, like on Yelp.
- We split the grid into four quadrants, creating a tree with four children (see Red, Green, Blue, Yellow)
- See that here's a lot of density in the blue cell!
- We specify a K value, saying that if any cell has more than K items, we recursively split it
- So we split it again!
- See again that our lower left cell has more than K=5 items, so we recursively split it again.
- **==Now when we want to find a particular business, we work our way down the tree accordingly, and this tree is the index that ends up being stored on disk.==**


## R-Trees
- [[R-Tree]] are a similar concept to Quad Trees, but instead of **crudely** splitting the world into even fourths, it does something more dynamic, trying to cluster the locations/businesses that are close to eachother, and then each of these larger groupings can even have a little overlap with eachother.
![[Pasted image 20250520173950.png]]
- It's the same general idea where we create this tree and put it on disk, then traverse the tree.
- It's fairly complex and not something that we'll go into in detail in this video# Wrap


# Geospatial Index Commentary
- [[Geohash]] is very popular today; it exists in things like Redis; it's the default... many production databases rely on Geohashing, relying on B-Trees.
- [[QuadTree]] were foundational to the development of geospatial indexes, but aren't used much.
- Instead, [[R-Tree]], their successor, are more often used in production. For instance [[PostgresDB|Postgres]] uses R-Trees!
- In a SD interview, if you get tasked in a situation where you have two-dimensional lat/lon data, you know you want to mention some geospatial index, and mention to your interviewer that you understand the differences between these, and that you want to specify a particular one.


# DB Indexing: Inverted Indexes
- Another place where B-Trees might not be a good choice for you DB index:
![[Pasted image 20250520174354.png]]
- If we want to select all businesses with Pizza in the name... 
- Remember that in B-Trees, if we have an index on the name field, then our names are sorted lexicographically!
	- This would be great if we were doing a prefix search: `prefix%`, but if we're doing a search like above, we're screwed, and have to do a ==**full table scan!**== 💀

Instead, we should be using an [[Inverted Index]]!
- We can create a hashmap mapping each of the words that appear to all of the documents that they appear in!
- If we're searching for "fast", these documents in the lists are pointers to the pages that these obviously exist in on disk, and so we pull those pages into memory and return the relevant rows.
- These are used in [[ElasticSearch]], as well as [[PostgresDB|Postgres]]'s [[Full-Text Search Index|Full-Text Search]] feature.


# Note
- It's not likely in your interview that you'll get deep into the implementation of each of these indexes. It's more important that you understand where your queries might be inefficient, which columns you might apply indexes to, and, depending on the case, if there's a specific index to be applied.
![[Pasted image 20250520174744.png]]

