---
aliases:
  - Protocol Buffer
---


JSON schemas are flexible, but they require us specifying every field name in each message! This is inefficient if each message ha some preset scehma. If we have a predefined schema that all readers and writers are aware of, we can save space and maximize network throughput by omitting field names in place of a more efficient binary representation!


