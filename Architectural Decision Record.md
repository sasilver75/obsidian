---
aliases:
  - ADR
---
A short, dated document that ==captures one architectural decision, the context that forced it, and the consequences of choosing it.== The unit is the *decision,* not the feature or the system.
- c.f. with [[Specification|Technical Design Doc]]/Spec.
	- A spec might *contain* the rationale for a decision, but an ADR *is* the rationale, extracted so that it can be found later by someone who only cares about that one question.  
	- Typically they live in `docs/adr/0001-record-architecture-decisions.md`, `0002-...`, etc., ==checked into the repo next to the code they govern!==

The canonical template is tiny:
```
Date: 2026-04-12
Status: Accepted

## Context
We need vector search for the obligation evaluator. Corpus is <100k
embeddings, already running Postgres, team has no ops budget for a
second datastore.

## Decision
Use pgvector with HNSW indexes in the existing Postgres instance.

## Consequences
+ One datastore, one backup story, transactional consistency with rows.
+ No new vendor, no new SDK.
- Caps us around ~10M vectors before we'd need to revisit.
- HNSW build times will hurt if we ever bulk-reindex.
```

Typically immutable once accepted, append-only, with the following status lifecycle:
- ==Proposed==: Under discussion
- ==Accepted==: Decided, in effect
- ==Deprecated==: No longer the recommended approach, but not actively replaced.
- ==Superseded by ADR-NNNN==: Replaced; You don't edit the old ADR, you write a new one and link back. The history is the point.

#### What makes a good ADR?
- One decision per record.
- Context section names the forces, not just the situation. What constraints made this hard? What did we want that we couldn't have?
- Alternatives considered, with why not. This is the part that future-you actually needs.
- Consequences include the bad ones. If you can't name a downside, you didn't actually make a decision, you made a wish.
- Short! A page! If it's longer, it's a design doc with an ADR buried in it.


