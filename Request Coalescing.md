Generally, a concurrency technique where multiple simultaneous requests for the same resource are collapsed into a shared in-flight operation, where instead of every request doing the same expensive work, *the first* request performs the work, and the others wait for its result and reuse it.

A term typically referring to the way in which we can avoid [[Cache Stampede]]s, which take down services.


## Example Scenario: Cache Stampede avoidance
Imagine that a highly-requested Cache key (usually receives 10,000 requests per second) becomes stale because its [[Time to Live|TTL]] expires.
- Suddenly, 10,000 requests get a cache miss
- Each of these requests (in the case of a [[Cache-Aside]] pattern) would then make a request to the backing database  for that data.
- This can be very expensive, and take up all the resources of the data, causing requests to fail, retry, fail, etc.

**Instead**, with request coalescing, just *the first cache miss* makes the request to the backing datastore to get/recompute the value to store in the cache. The other requests *wait* for that data to become available in cache, and then serve a response from there.

Implementation-wise, this is usually implemented as a per-key "in-flight registry":

```
GET key
	cache hit -> return value
	cache miss ->
		if someone is already loading key:
			wait for their result
		else:
			mark the key as loading
			fetch from DB
			write data to cache
			wake waiters
			clear loading market
			return value
```
Re: "wake waiters": In code, "waiters" might be blocked on a Promise, Future, condition variable channel, semaphore, etc. They're not asleep in a literal sense, they're just suspended/blocked/awaiting the result of a shared operation.

1. In a single-app-server scenario, this can just be implemented with an in-memory map of `key -> Promise/Future`
2. Across multiple app servers, you need distributed coordination, usually something like a short-lived [[Redis]] lock.
	- `SET lock:homepage_feed random_token NX EX 5`
		- This means "Set key `lock:homepage_feed` to `random_token`, but only if the key does not already exist, and automatically expire it after 5 seconds."












