---
aliases:
  - Thundering Herd
---


When many requests hit a [[Cache Miss]] for the same popular key at the same time (e.g. because its [[Time to Live]] has just expired), causing them all to fetch or recompute the value from the backing database, which can often bring its availability down and cause requests to fail, retry, etc..

Example:
- homepage_feed cache entry expires
- 100,000 requests arrive
- all see cache miss
- all query the database
- database gets overwhelmed

Common mitigations include [[Request Coalescing]], [[Cache Warming]], and probably [[Stale-While-Revalidate]] too (since that lets you basically still serve stale data after an entry is expired, up to a limit).

