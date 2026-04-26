---
aliases:
  - JSONL
---
Ref


A text format, often used for large datasets, logging, and LLM training, where each line is a valid [[JSON]] object separate by a newline character.

```JSON
{"id": 1, "name": "Alice", "role": "engineer"}
{"id": 2, "name": "Bob", "role": "designer"}
{"id": 3, "name": "Carol", "role": "manager"}
```

Commonly used for LLM training in a chat format:
```JSON
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]} 
{"messages": [{"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris"}]}
```



