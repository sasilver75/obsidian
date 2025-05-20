SDIAH Elastic Search Deepdive: https://www.hellointerview.com/learn/system-design/deep-dives/elasticsearch

ElasticSearch is one of the most popular and powerful open-source search engines.
- Understanding ElasticSearch is often relevant for interviews where search isn't even the primary functionality, and for certain types of interviews, like the Product Design interview at Meta.

First, let's talk about Search -- what is Search?
![[Pasted image 20250520131756.png]]
Let's use a Bookstore as a motivating example throughout the video:
- Let's say we want to search the book store for `Title=Great,Price<13` ==criteria==. 
- We might also want to ==sort== the results in a certain way, such as sorting by publish date.
- In the search results page, we might also have a way to break down the search and refine what we've described... these are typically called ==facets/faceted search==. For example, we might want to refine by the publisher.
- The net result of a search experience is ==results==; some (ordered) set of objects.


So how does ElasticSearch think about Search?
![[Pasted image 20250520132218.png]]
We should name some concepts:
- ==Document==: Not necessarily PDFs or things like that; just JSON blobs that can contain anything.
- ==Index==: An overloaded term in computer science... here, it refers to a collection of documents that we want to make searchable.
- Mapping: In an Index, a Mapping specifies the ~schema that we want to be searchable on our docuemnts.
	- Our documents might have dozens of fields, but our users might not need to search by all of these.
	- The ==Mapping== tells our search engine which of these fields are pertinent with respect to search, and what their types (==Fields==) are. If ElasticSearch knows that Price is a Float, it can prepare to handle range queries.

So how do we bring documents into ElasticSearch and make them searchable?
First, we need to ==create an index==:
- ElasticSearch has a pretty good [[REST]]ful API
![[Pasted image 20250520132343.png]]
- Here, first we're doing a PUT to our /books **Index**, with some parameters. This creates our `books` index.
- We don't HAVE to specify a **Mapping**, but we CAN. The reason why we might want to is that if ElasticSearch can't infer the types of fields and which ones we might want to search, then it might make some mistakes. There's also some overhead in having ElasticSearch guess the fields that you might want to search on, since it might do more than you need.
	- Note that we're using float, text and keyword types
		- **float** lets use do range queries
		- **text** allows us to do full-text search
		- **keyword** is opaque; each of the tokens is a unit... appropriate for discrete categories. 
	- Note that we're including this nested "**reviews**" object... we'd like to be able to answer our search queries with as few requests as possible, and by nesting reviews in a document, we can answer our search queries with as few requests as possible, and by nesting our reviews in the document, we can basically construct queries like: "I want specific types of books having certain reviews"
		- If reviews are changing a lot or if my queries rarely touch these reviews, this will be quite expensive, and we might be better served by creating a separate index for them.

Let's talk about ==adding== **Documents** to our **Index**. See the request and response for a POST request:
![[Pasted image 20250520140059.png]]
- Note that we POSTed the new book to the /books/\_doc endpoint
- When we do this, ElasticSearch responds with... a success message that tells us a number of details:
	- What the documents identifier is, what index it was inserted into, and its **version number!**
		- It's important that we be able to handle collisions and concurrency; this is particularly relevant when we start to talk about updates.

So let's talk now about updating **Documents**:
![[Pasted image 20250520140325.png]]
We pass in the entire document body to update it
- This might be overkill if we just want to update the price.
- If two updates are submitted simultaneously, the last one wins!
	- So in the top-right, we ==can== pass in a VERSION NUMBER, so that we'll reject the update if the version number doesn't match. This gives a chance the handle the collision/version number on the client, who will query the latest version number, take a diff, decide what to do, and resubmit the request.
	- This is a ==common pattern== in distributed systems wher we want to make sure that we're updating what we think we're updating.
- Note in the bottom right that we can also POST to the \_update endpoint to simply bump the price from 13.99 to 14.99. I assume that here's also a version=? query param that we can pass.


So let's talk about ==querying in ElasticSearch!==
![[Pasted image 20250520140549.png]]
- There's a nice /indexname/\_search that lets you query with a JSON-like query language.
	- (1) See that we're doing a match query on title=Great.
	- (2) We can extend that by saying that we ALSO want it be $15
	- (3) We're querying the books index where the reviews have a comment of "excellent" and the rating is GTE 4/5.
- Note that these are GET endpoints; you CAN still put this in the body of the request, or you can stuff them into the Query parameter, but you'll be limited by the size of a URL, which is ~2048.

Once we've got Search ==Results==, they're going to come back in something that looks like this:
![[Pasted image 20250520140846.png]]
- See that ES is telling us some metadata about the search process itself, as well as some hits, which are the results.
- In some case you might want to not return the entire document; it's possible to only return specific fields from the document (e.g. the title of the book). 
- Note also that we get a ==score== for each hit, which tells us the relevance of the item to our search query!

ElasticSearch has the ability to let you ==specify your sort order== of returned items easily:
![[Pasted image 20250520141105.png]]
- (2) Sort on multiple keys
- (3) An ES DSL named "painless" that lets you specify basically formulas that you can use to calculate sorting orders.
- (4) Can also sorted by those nested objects/fields; Here, we're sorting the returned books by the highest review.

ES lets us specify the sort order very rigidly; if we want to, we can tell it exactly which fields to order on, or we can use that \_score field from the result set, which orders by ==relevance==. 
- Some of the simplest algorithms are pretty intuitive to understand. One of the simplest to understand is [[TF-IDF|Term Frequency Inverse Document Frequency]] (TF-IDF). 
	- The most relevant document probably contains my terms more frequently!
	- Favor those words that don't appear very frequently across all documents!
![[Pasted image 20250520141521.png]]
- Above:
	- Those items which contain keyswords that are densest and don't appear in many other places higher than those keywords that don't appear very often, or appear in almost all the other documents in my corpus.



We need to be able to do some sort of ==pagination== for our results!
- There are two forms of pagination:
	- **Stateful Pagination**: Requires that the server keep track of something, so that it knows what the next page is. We're usually talking about Cursors here.
	- **Stateless Pagination**: We'll pass in some sort of parameter that lets us localize approximately where in a result set we'd like to look.
![[Pasted image 20250520141749.png]]
Two approaches for this **Stateless Pagination**:
- (1) We'll increment the from:10 and keep the size the same for the following query.
- (2) We can also use a sort after... so we're sorting by date and ID, and when we get to the bottom of our results, we have a date after which there are no more results in subsequent pages, and an id after which there are no more results.
	- This is saying "Don't return me any results that might be before these two quantities.
	- This ==Search After== technique is preferable, particularly in cases where you want to go deep into the pagination, since thee first option says "Get all of the results, then slice into the results based on the inputs", whereas this Search After technique lets us restrict the documents that we're querying.

==Both of these have a problem: The index is constantly changing!==
- New documents are being added, deleted, etc.
- In the ==worst-case scenario==, you have a document that *would* have been on the next page, but because the index changed, it now belongs on the page that you're currently on. In that case, your pagination is going to MISS that document!
- How do we deal with that?
	- ![[Pasted image 20250520142121.png]]
	- ElasticSearch supports a =="Point in Time" search==, where ES will snapshot what the index looks like at that period.
		- This returns an ID, and then we can use that PIT ID when we're searching!
		- Note that we specify a keep-alive time; we're specifying in our POST that we still want to keep it for 1 minute, every time we search.



## How to use ElasticSearch in your System Design Scenarios!
1. Typically for complex search scenarios; not your primary transactional database
	1. [[Full-Text Search Index|Full-Text Search]], [[Geospatial Indexes]], [[Vector Search]]
	2. Often used together with a primary transaction database (e.g. [[PostgresDB|Postgres]], and then use [[Change Data Capture]] to eventually-consistently move writes into the search index.)
2. Best with read-heavy workloads
	1. If you've got a lot of writes happening, and writes dominate reads, then ES might not be appropriate. There are a lot of ways to change the nature of your workload (e.g. limiting the number of writes, batching them out), but generally speaking, ES performs best when you have more reads than writes.
3. Must tolerate eventual consistency
	1. It's endemic to the design of ElasticSearch. If you need [[Strong Consistency]], you're going to need to use a different database. ((I think they're saying that even if you had a primary transactional database and where doing 2PC to keep them consistent))
4. Denormalization is key
	1. You shouldn't have do joins in ElasticSearch. **==JOINS ARE NOT SUPPORTED IN ES==**, and the way you might bolt-on joins by doing lookups outside of ES will compromise performance.
5. Do you even need it?
	1. ES clusters aren't trivial to operate and maintain; A simple search capability in your [[PostgresDB|Postgres]] database might work, unless you need full-text search over billions of documents.
	2. ==If you have simplistic needs, use simplistic solutions!==


In a lot of interviews, you need to point out when you might use ElasticSearch and when you might use it in your overall design. If you're designing Twitter, search might be a part of your design, and delegating it to ES might make sense. In other interviews, your interviewer might explicitly tell you "Don't use ES, I'd love to see how you implement some of the core concepts." Being able to understand what goes on beneath the covers of ES is what this second section is about.

The first thing to understand about ES is that it's something of an orchestration for [[Lucene]], which is a low-level search functionality.
- ElasticSearch handles:
	- Distributed System Aspects
	- API
	- How to manage nodes, backups, replication
- Lucene:
	- Tries to make sure the data is quickly searchable
	- Operates on a single node.


## Let's peel back the layers on ElasticSearch, and talk about its internals!

![[Pasted image 20250520143431.png]]
Note that an ElasticSearch cluster can have a single server! **Node** isn't necessarily a physical machine!
But at a conceptual level, ==these are the responsibilities that exist inside an ES cluster that need to be performed by some hardware somewhere.==
- ==Master Node==: When we spin up our cluster, we specify seed nodes/hosts that ES will try to connect to, and then will **elect** a master of the cluster. We only want one master of the node so that it can make **administrative decisions** like "Which nodes are part of the cluster?" There's only one master elected at any one point in time. 
- ==Coordinating Node==: Think of this as the **API layer** to the cluster. The vast majority are Search requests, and this node takes the request, parses them, and passes them to the relevant nodes that have the data that we need in order to return the search results. Users are, for the most part, interacting with these.
- ==Data Node==: Where are indexes are stored; where  we keep our documents. We're going to have a lot of these. Going to have a lot of I/O, so this will need a lot of memory or some pretty fast disk.
- ==Ingest Node==: The backend for our service. How the documents make their way into the cluster.
- ==Machine Learning Node==: The ML nodes are not always used, but are used for ML-intensive tasks; these nodes might require access to a GPU, and are going to have very different access patterns.

Notice above that **some of these require different hardware!**
- Master nodes need to be robust and reliable
- Coordinate nodes will do a lot of network traffic with the outside world
- The Data Nodes will need a lot of I/O-helping memory/fast disk
- Ingest Nodes will do a bunch of analysis of documents, and are typically more CPU bound.
- We mentioned that ML noes might need a GPU.
**==So when we set up a cluster, we might set up different server/VM types for these different node responsibilities to optimize our cluster!==**


![[Pasted image 20250520144832.png]]
We mentioned before that the fundamental grouping of ES clusters are **Indexes**
- Inside of an **Index**, we have [[Sharding|Shard]]s and [[Replication|Replica]]s.
	- These Shards are resposnible for containing all of our documents and building up indexing structures to make the searches fast.
	- The Shards are mutually exclusive, in the sense that if we have a Document, it will be assigned to a specific Shard.
	- The Replicas are basically carbon copies of these Shards.
	- We might have Replicas of a Shard across multiple machines!
	- So when a request comes in, the **Coordinating Node** can decide whether it wants to use Shard 1A or Shard 1B to serve a certain read!
	- Shards have a hard limit of ~2B documents, which is a lot, but we still might want to divide those documents because it's too big for a single machine or if we want to spread the load across the cluster, to make it faster!
- Inside of each Shard are these [[Lucene]] Indexes.
	- These Lucene Indexes are 1:1 with ES Shards.
	- A Shard is basically just encapsulating a Lucene Index.
	- Inside a Lucene Index we have **Segments**.
	- A Lucene Index accumulates these **immutable segments**.
	- When we create a document, Lucene will try to batch that document into as many writes as it can contain before flushing it out to contain a segment with multiple documents.
	- When we add more documents later, we might create a new segment!
	- At some point in the future, Lucene may merge these segments into a new segment, and delete the old ones, a process called [[Compaction]].
- So how do updates work in ElasticSearch, from a Lucene Perspective?
	- For updates, we're actually doing a Soft Delete!
	- Lucene, in these segments, also contains a mapping of deleted documents. When we later query that segment, we remove results that contain deleted items.
	- When we do an update, we mark the document as deleted in the old segment and then create the document in the new segment.
	- This maintains this **immutability** property; aside from the map of deleted documents, these segments aren't changing. This enables us to.. exploit a bunch of caching and concurrency benefits. Since Segments aren't necessarily changing, when we load them into memory, we can do them in a compressed format and load them all at once, guaranteeing that they aren't changing underneath me. Can also have multiple readers touching the data in a segment in a concurrent way without worrying about race conditions or contamination! The segment data won't change; it's read-only, in most respects. 

Inside the Lucene Segments,we need to store all of our documents.
- Conceptually, you can think of a Segment "containing" documents, but these searches need to be fast; it's not enough to just dumbly store the data in our segments, we need to be able to search quickly!
- Lucene uses several techniques to pull them off, one of which is an [[Inverted Index]].
![[Pasted image 20250520145756.png]]













