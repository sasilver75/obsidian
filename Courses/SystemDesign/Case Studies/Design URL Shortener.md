
Bit.ly is a URL shortening service that converts long URLs into shorter, manageable links. It can also provide analytics for the shortened links.

We'll do the usual set of steps:
1. Functional Requirements
2. Non-Functional Requirements
3. Back-of-the-Envelope
4. Core Entities
5. APIs
6. Data Paths
7. High-Level Design
8. Low-Level Design


### Functional Requirements
- Let's focus on these:
	1. Users should be able to submit a ==long URL== and receive a ==short URL==.
		- Users should be able to specify a ==custom alias== for their shortened URL.
		- Optionally, users should be able to specify an ==expiration date== for their shortened URL.
	2. Users should be able to access the original URL by using the shortened URL.

### Non-Functional Requirements
- These types of requirements often define system attributes like scalability, latency, security, and availability, and are often framed as specific benchmarks.
- Core Requirements:
	- The system should ==ensure uniqueness== for the short codes
	- ==The redirection should occur with minimal delay==
	- The system should be reliable and available 99.99% of the time (==availability > consistency==)
	- The system should scale to support 1B shortened URLs and 100M DAUs

An important consideration here is the significant ==imbalance between read and write operations==. The read-to-write ratio is heavily skewed towards reads, as users frequently access shortened URLs, while the creation of new short URLs is comparatively rare.


### Defining the Core Entities
- We recommend that you start with a broad overview of the primary entities.
- At this stage, it's not necessary to know every specific column or detail, we'll focus on that later.
- In a URL shortener, the core entities are straightforward:
	- Original URL
	- ShortURL
	- User

That's all we need for now!

### API
- Now let's define the API of the system. Typically here we just go one-by-one through the core requirements and define the APIs that are necessary to satisfy them.
	- Usually, these map 1:1 to a functional requirement, but there are times when multiple endpoints are needed to satisfy an individual functional requirement.
- We know from our functional requirements that we basically just need to be able to:
	1. Create a ShortURL from a LongURL (optionally with custom shortURL, expirationDate)
	2. "Get" a ShortURL, getting redirected to the LongURL
- We'll typically assume that we'll use REST.
	- So we'll use POST and GET for our two requirements.

For the "Create a shortURL (with optional custom shortURL and expirationDate)"
```
POST /urls
{
	longURL
	shortUrl?
	expirationDate?
}

->
{
	longUrl
	shortURL
	expirationDate?
}
```

For the "GET a shortURL and get redirected to a longURL"
```
GET /urls/{shortUrL}

-> HTTP 302 Redirect to longURL
```

### High Level Design
- Now, we can again go through our functional requirements and designing a single system to satisfy them. Once we have this in place, we'll layer on depth for our deep dives!

First functional requirement: User should be able to submit a longURL and receive a shortened version.
- The core part of this is figuring out how we're going to generate a short URL.

At the highest level, we need:
- A client that makes requests
- A server the is able to generate the shortURL and store it in...
- A database
	- Urls table
		- id
		- longURL
		- shortURL
		- expirationDate
		- createdBy
	- Users

This might look like:
![[Pasted image 20260621134651.png]]

When a user submits a longURL, the client sends a POST request to `/urls` with the long url, custom alias, and expiration date. Then:
1. The Primary Server receives the request and validates the longUrl format. 
	- We can optionally can check if the exact longURL was already created, and return the existing shortUrl. Most URL shorteners allow multiple shortUrls for the same longUrl though, since different users may want separate expiration dates, independent analytics, or different custom aliases.
2. If the URL is valid, we generate a short code.
	- For now, we'll abstract this away as some magic function that takes the URL and returns a shortURL. We'll dive deeper into this later.
	- If the user has specified a custom alias, we can use that as a short code, after validating that it doesn't already exist.
		- To prevent custom aliases from colliding with future counter-generated codes, consider prefixing generated codes wit ha character that custom aliases can't use, or store them in separate namespaces.
3. Once we have a shortUrl, we can proceed to insert it into our database, storing the short code (or custom alias), long URL, and expiration date.
4. Finally, we can return the shortUrl to the client.


The second functional requirement is that users should be able to access the shortenedUrl. Importantly, this shortUrl exists at a domain we own! Something like `short.ly/abc123`.
![[Pasted image 20260621135151.png]]
The flow looks like:
1. User's browser makes a `GET short.ly/abc123`
2. The Primary Server looks up a `Urls` record in the database having that shortURL
3. If the short code is found and hasn't expired, return the longUrl with a redirect. For expired URLs, return a `410 Gone` status.
	- For cleanup, we run a background job periodically to delete expired rows from the database.
4. The server then sends an HTTP redirect response to the user's browser, instructing it to navigate to the original long URL.
	- There are two main types of HTTP redirects that we could use:
		- 301 (Permanent Redirect): Indicates that the resource has been permanently moved to the target URL. Browsers typically cache this, which means that our server might be bypassed in the future.
		- ==302 (Found):== Indicates the resource is temporarily located at a different URL. Browsers don't cache this response, ensuring that future requests for the shortURL will always go through our server first.
			- This gives us more control over the redirection process, allowing us to update or expire links as needed. It prevents browsers from caching the redirect, which could cause issues if we need to change or delete the shortURL in the future. It also lets us track client statistics.for each short URL.

### Deep Dives
- At this point, we have a basic, functioning system that satisfies the functional requirements.
- We need to look back at our *nonfunctional requirements* and see which ones still need to be satisfied or improved upon.

1. How can we ensure short URLs are unique?
	- Bad Solution: Taking a prefix of our LongURL (collisions are likely as we get more URLs)
	- Great Solution: We need some entropy to ensure that our codes are unique; we could try a hash function like [[SHA-256]] to generate a fixed-size hash code. We can then take the output and encode it using a [[Base62]] encoding scheme and take just the first N characters as our short code. Still, there's a chance of collision. To handle collisions, we can implement a `UNIQUE` constraint on the short code column and retry with bounded (e.g. 3-5) retries before falling back to a different strategy or returning an error.
	- Greatest Solution: Unique Counter with a Base62 encoding. We simply increment a counter for each new URL, and then take the output of the counter and encode it using a [[Base62]] encoding to ensure it's a compact representation. [[Redis]] is good for this, because it's single-threaded, eliminating race conditions. Each counter value is unique, eliminating the risk of collisions. Note that simply encoding it doesn't really "hide" it from users, which makes things like a malicious user scraping all of our shortUrls straightforward. There are libraries that do a [[Bijective]] mapping from the counter to a Base62 string that we can use to get around this.
2. How can we ensure that redirects are fast?
	- Without optimization of any sort, taking a shortURL and finding the corresponding longURL would be a full table scan, which can get slow with billions of records.
	- Good solution: Adding an Index on shortURL (if it's our PK anyways, an index is automatically created)
	- Great solution: Implementing a [[Cache]] (e.g. Redis). We can introduce an in-memory cache like redis or Memcached between the application server and the database, perhaps in a Cache-Aside pattern.
		- ((I think a Cache-Aside pattern would be fine. My thought is that a single cache miss doesn't matter too much, so we don't have to think about things like pre-warming or stale-while-revalidate. Because I imagine the read pattern to be pretty lopsided for a few cache items, most reads will hit a warm cache entry anyways. If someone were to ask about eviction, I'd say LRU. We don't need to think too much about invalidation because we don't have any sort of update path.))
	- Great solution: Leveraging a [[Content Delivery Network|CDN]]: The shortURL domain can be served through a CDN with points of presence geographically distributed around the world. The CDN nodes themselves cache the mappings of short codes to long codes. We can even deploy the redirect logic to the edge using platforms like [[Cloudflare Workers]]s or [[Amazon Lambda|AWS Lambda@Edge]], so that the redirection can happen directly at the CDN level without reaching the origin server. This too prevents some challenges! Ensuring cache invalidation and consistency across CDN nodes can be complex... CDNs and edge computing might also incur higher costs.
3. How can we support 1B shortened URLS and 100M DAUs?
	- We've talked about scaling reads (and their latency) by introducing a cache, but how can we scale writes?
	- Each row in our database consists of ~200 bytes.
		- Short Code: 8 bytes
		- Long Code: ~100 bytes
		- customAlias: ~100 bytes
		- expirationDate: 8 bytes
		- creationTime: 8 bytes
	- So if we had 1B mappings, that's 500 bytes * 1B rows = 500GB of data
	- That's a single instance, no need to shard or anything like that.
	- Most database technology will work here; we've offloaded the heavy read through to a cache and write throughput is pretty slow.
	- For high availability....
		- We can consider [[Replication]], if only for [[High Availability|HA]] so that we can redirect to another if one server goes down.
		- We could also implement a backup system that periodically takes a [[Snapshot]] of our database and stores it in a separate location. This adds complexity to our system design, as now we need to ensure that our Primary Server can interact with the backup without nay issues.



Some people might talk about separately splitting out a read service and a write service, on account of hte significant skew towards read workloads.

![[Pasted image 20260621150233.png]]
This would allow us to horizontally scale (e.g.) our read service, which would receive much more traffic... but it might not be worth the operational complexity.















