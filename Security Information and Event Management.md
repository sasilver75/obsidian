---
aliases:
  - SIEM
---



```
Sources (firewall, WAF, IDS, EDR, cloud logs, identity)
         │
         ▼
       SIEM       ← aggregates, correlates, generates alerts
         │
         ▼
       SOAR       ← runs playbooks: enrich, contain, ticket, remediate
         │
         ▼
     Analyst      ← only sees the alerts that survive automation
         │
         ▼
     Resolved
```