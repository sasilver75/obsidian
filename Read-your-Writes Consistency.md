---
aliases:
  - RYW Consistency
---
A [[Consistency]] model.

A client always sees its own writes, even if other clients may temporarily see older data.

# How it is achieved
- The client or session tracks the latest version it ==***wrote***==, such as a log sequence number, timestamp, vector, or session token.
- Future reads include that token. the system then either routes the read to the leader, or routes it to a replica that has applied at least that version, or waits until a chosen replica catches up.
- [[Sticky Session]]s are a simpler version of this: keep the user reading from the same node that handled their write.

# Tradeoffs
- Requires session tracking. Can reduce load-balancing flexibility. A user may see their own write, while others still do not, which can be a positive or negative/confusing thing depending on the context.

# Use Case
- This is commonly used in social media platforms where users expect to see their own profile updates right away, or when they expect to see their own post on the timeline. Editing your profile, changing settings, posting a comment, saving a draft.
- You mainly need the author to see their own change immediately. Other users seeing it a few seconds later may be acceptable.

