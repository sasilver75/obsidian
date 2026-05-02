---
aliases:
  - Spec
  - Design Doc
  - Design Document
  - RFC
  - Technical Design Doc
  - TDD
---
Question it answers: *How will we build it?*
- Owned by **engineering**
- Audience: other engineers; primarily reviewers and future-you trying to understand decisions.
- Assumes the PRD's "what and why", and gets concrete about ==data model==, ==APIs==, ==components==, ==state machines==, ==migration plan==, ==failure modes==, ==rollout==, ==observability==.
- Includes ==tradeoffs== considered and rejected, with reasons.
- Lives at the "system" or "component" altitude.

Typical sections: context, proposed design, alternatives considered, data model, API surface, migration/rollout, risks, open questions.

# [[Specification|Spec]]/[[Specification|Design Doc]] vs [[Product Requirements Document]]
- Specification/Design Doc: *How will we build it?*
- Product Requirements Doc: *What are we building, for whom, and why?*
- Relationship: PRD -> Spec -> Code

See also: [[Architectural Decision Record]]

