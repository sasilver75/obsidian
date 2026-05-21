---
aliases:
  - CoW
---
An optimization technique where ==multiple callers share the same resource== (memory, file, or data structure) initially, avoiding immediate, costly duplication..

A ==private copy is only created when one party attempts to modify the data==, ensuring efficient resource management and improved performance, particularly for large datasets.
- When copying data, the system creates a new reference rather than a full copy. The actual duplication occurs only when a "write" (modification) occurs.



