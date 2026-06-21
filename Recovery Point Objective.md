---
aliases:
  - RPO
---
The maximum amount of data loss that a system can tolerate, measured backward in time.
- "At most 5 minutes of writes," measured in time before the incident.
- RPO is about **data freshness.** RTO limits data loss.

If a database crashes at 12:00 PM, and if the system has an RPO of 15 minutes, then restoring to data from 11:45 AM is acceptable, but restoring to data from 10:00 AM violates the Recovery Point Objective.

Compare with [[Recovery Time Objective]] (RTO)




