---
aliases:
  - Eventually Consistent
---


The system will become consistent over time, but may temporarily have inconsistencies (so as to provide better [[Availability]]). This is the most relaxed form of consistency, and is often used in systems like DNS where temporary inconsistencies are acceptable.

This is the default behavior of most distributed databases and what we are implicitly choosing when we prioritize availability.