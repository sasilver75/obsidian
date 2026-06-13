A cache refresh strategy where ==frequently-used cache entires are proactively refreshed before they expire, usually in the background==, so later reads can keep hitting fresh cached data without waiting for a reload.

It trades extra background work for fewer cache misses and lower read latency.
- ==Use when a small set of known hot keys are read often, are expensive to recompute, and when you want to avoid users ever hitting the cache miss and experiencing the latency of a refresh.==

This is a policy that has to be implemented by application code, a cache library, a worker, etc. To refresh a cache entry, *something* must know both the cache key and the loader function for rebuilding it. Redis itself may know that `product:123` expires soon, but it doesn't know how to fetch product 123 from your DB and rebuild the cached JSON.

A common implementation is ==read-triggered==:
- read `product:123`
- cache hit
- entry is still fresh, but expires in 5 seconds
- return cached value immediately
- enqueue background refresh for product:123. which loads fresh data from the DB/API and writes a new cache value with a new TTL.
(Pretty similar to [[Stale-While-Revalidate]], mechanically, imo)

Another implementation is ==worker-driven==:
- A job periodically refreshes known hot keys:
	- Every minute:
		- Get top dashboard/product/feed keys
		- Refresh keys expiring soon
This seems more like what you'd expect, but of course this one doesn't seem to have a knowledge of when keys are going to expire.

_________

Similar in a sense to [[Stale-While-Revalidate]], in that both use background refresh to avoid making reads wait, the difference is when the refresh is triggered:
- In ==Refresh-Ahead==, it's more proactive; the cache decides to refresh before expiry, often based on TTL, access frequency, or scheduled refresh.
- In ==Stale-while Revalidate== is more reactive: a read request finds expired-but-servable data, returns it, and triggers a refresh.

Comparison to [[Read-Through Cache]]:
- Refresh-Ahead proactively populates the cache with data from the backing store *before* it is explicitly required.
- Read-Through Caches fetch data from the backing store only when it is explicitly requested by the application.

________________

Minimal Example in Python:
```python
import json
import threading
import time

import redis


r = redis.Redis(host="localhost", port=6379, decode_responses=True)

CACHE_TTL_SECONDS = 300
REFRESH_AHEAD_SECONDS = 60


def refresh_cache(key, load_value):
	# Given a load_value(...) callable, call it, and store the value @ key
    value = load_value()
    r.set(key, json.dumps(value), ex=CACHE_TTL_SECONDS)


def get_cached(key, load_value):
	# Get the value from the cache (here, a json-formatted string)
    cached = r.get(key)
	
	# If we didn't get a value, load it from the DB, set it (with TTL) and return.
    if cached is None:
        value = load_value()
        r.set(key, json.dumps(value), ex=CACHE_TTL_SECONDS)
        return value

	# Get the TTL for the key that we retrieved
    ttl = r.ttl(key)

	# If it's time to refresh that key... kick off <some sort of process> to update it
    if 0 < ttl <= REFRESH_AHEAD_SECONDS:
        thread = threading.Thread(
            target=refresh_cache,
            args=(key, load_value),
            daemon=True,
        )
        thread.start()

	# Return the object version of the cache item
    return json.loads(cached)


def load_user_profile():
    time.sleep(0.25)  # Simulate database/API latency.
    return {
        "name": "Ada Lovelace",
        "loaded_at": time.time(),
    }


profile = get_cached("user:123:profile", load_user_profile)
print(profile)
```


Complete Example in Python, adding:
- Redis refresh lock: Prevents many workers from refreshing the same cache key at the same time.
- Unique lock token: Prevents one worker from accidentally deleting another worker’s newer lock.
- Lua unlock script: Makes “check that I own the lock, then delete it” an atomic Redis operation.
- Thread pool: Limits background refresh concurrency instead of creating an unbounded number of threads.
- Error handling: Keeps cache reads from failing just because a background refresh failed.
- Lock Time To Live: Ensures a crashed refresher does not leave the refresh lock stuck forever.
- Reusable `fetch_value` function: Lets the same refresh-ahead helper work for different cached values, such as users, products, permissions, or API responses.
- Separate cache Time To Live and refresh-ahead window: Lets the value live for one duration while refreshing only near the end of that duration.
```python
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

import redis


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

executor = ThreadPoolExecutor(max_workers=8)

CACHE_TTL_SECONDS = 300          # Cached value lives for 5 minutes.
REFRESH_AHEAD_SECONDS = 60       # Refresh when <= 60 seconds remain.
LOCK_TTL_SECONDS = 30            # Refresh lock expires if worker crashes.


RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

release_lock = redis_client.register_script(RELEASE_LOCK_SCRIPT)


def fetch_user_profile_from_database(user_id: str) -> dict[str, Any]:
    """
    Simulated slow source of truth.

    In a real system this might query PostgreSQL, call an internal service,
    or request data from an external API.
    """
    time.sleep(0.25)

    return {
        "user_id": user_id,
        "display_name": "Ada Lovelace",
        "fetched_at": time.time(),
    }


def set_cached_json(key: str, value: Any, ttl_seconds: int) -> None:
    redis_client.set(
        key,
        json.dumps(value),
        ex=ttl_seconds,
    )


def try_refresh_in_background(
    key: str,
    fetch_value: Callable[[], Any],
) -> None:
    lock_key = f"lock:refresh:{key}"
    lock_token = str(uuid.uuid4())

    acquired = redis_client.set(
        lock_key,
        lock_token,
        nx=True,
        ex=LOCK_TTL_SECONDS,
    )

    if not acquired:
        return

    def refresh() -> None:
        try:
            fresh_value = fetch_value()
            set_cached_json(key, fresh_value, CACHE_TTL_SECONDS)
        except Exception:
            # In production, log this exception.
            # The old cached value remains until its TTL expires.
            pass
        finally:
            release_lock(keys=[lock_key], args=[lock_token])

    executor.submit(refresh)


def get_refresh_ahead_json(
    key: str,
    fetch_value: Callable[[], Any],
) -> Any:
    cached_value = redis_client.get(key)

    if cached_value is None:
        fresh_value = fetch_value()
        set_cached_json(key, fresh_value, CACHE_TTL_SECONDS)
        return fresh_value

    ttl = redis_client.ttl(key)

    if 0 <= ttl <= REFRESH_AHEAD_SECONDS:
        try_refresh_in_background(key, fetch_value)

    return json.loads(cached_value)


def get_user_profile(user_id: str) -> dict[str, Any]:
    key = f"user:{user_id}:profile"

    return get_refresh_ahead_json(
        key=key,
        fetch_value=lambda: fetch_user_profile_from_database(user_id),
    )


if __name__ == "__main__":
    print(get_user_profile("user_123"))
```