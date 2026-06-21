---
aliases:
  - MTTR
  - Mean Time to Restore
---
Average time it takes to restore a failed service to an acceptable working state after an incident begins.

```
Failure happens
   ↓
Incident detected
   ↓
Incident acknowledged
   ↓
Cause identified
   ↓
Mitigation applied
   ↓
Service restored
   ↓
Incident fully resolved
```

"Followed by" [[Mean Time to Failure]]


Relation to [[Recovery Time Objective|RTO]]:
- Recovery Time Objective (RTO) is a target: "We need this service restored within 30 minutes."
- Mean Time To Recovery (MTTR) is an observed average: "Historically, this service takes 42 minutes to recover."
- RTO is the goal, and MTTR is measured performance. If MTTR > RTO, the system or operating process is not meeting the business requirement.



