---
aliases:
  - RTO
---
The maximum amount of downtime a system can tolerate, measured forward in time after a failure.
- "Service must be restored within 30 minutes after the incident."
- RTO is about **service availability.** RTO limits downtime.

If a database crashes at 12:00 PM, and the system has a RTO of 1 hour, then the system should be usable again by 1:00 PM. It doesn't necessarily mean that every underlying problem is fully-fixed by then; it means the business-critical function is restored enough to operate.

See also [[Recovery Point Objective]] (RPO)



