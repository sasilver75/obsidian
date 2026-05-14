---
aliases:
  - Prompt Cache
  - Context Cache
---

Reuses a previously-processed prompt prefix across requests so the model skips re-encoding it, cutting latency and input-token cost on repeated calls.

- **How it works**: provider hashes the prefix (system prompt, tools, long context) and stores its computed KV state. Subsequent requests with the same prefix hit the cache and only pay full price for the new suffix.
- **Cache writes** cost more than a normal input token (e.g. Anthropic: 1.25× for 5-min TTL, 2× for 1-hour); **cache reads** are far cheaper (~0.1× input).
- **TTL**: short by default (Anthropic ~5 min, refreshed on each hit). Longer TTLs available at higher write cost.
- **Prefix must match exactly** — even a whitespace change invalidates. Put stable content (system prompt, tool defs, RAG context) first; put volatile content (user turn) last.
- **Good fits**: agent loops, long system prompts, multi-turn chat, repeated RAG over the same corpus, eval suites.
- **Bad fits**: one-shot calls, prompts that change every request, tiny prompts where the write premium outweighs reuse.

See also: [[KV Cache]]
