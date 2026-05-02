---
aliases:
  - SOAR
---
The category of tools that automate security operations workflows, especially on the ==response== side:
- When an alert fires, SOAR runs the playbook that a human analyst would otherwise execute by hand.

SOAR systems exist because [[Security Operations Center]]s (SOCs) are drowning in alerts -- tens of thousands a day from [[Security Information and Event Management|SIEM]]s, [[Intrusion Detection System|IDS]], [[Endpoint Detection and Response|EDR]], cloud logs, etec. The vast majority are [[Type One Error|False Positive]]s or low-severity, but each needs triage. Without automation, analysts spend their time copy-pasting IPs into VirusTotal, looking up users in Okta, asking IT to reimage a laptop, etc.
- ==SOAR codifies repetitive parts of SOC analysts playbooks and runs them automatically.==

Three words:
- ==Orchestration==: Coordinate across many security tools ([[Firewall]], [[Endpoint Detection and Response|EDR]], [[Security Information and Event Management|SIEM]], [[Amazon Identity and Access Management|IAM]], ticketing ,chat) via APIs.
- ==Automation==: Execute repetitive tasks without a human
- ==Response==: Actually act on incidents, not just observe them. Triage, contain, remediate.

Alert: "EDR detected suspicious PowerShell on laptop-042."

  A SOAR playbook might:
  1. Enrich — pull the user's identity from Okta, the laptop's last-known location from MDM, recent logins
  from the SIEM, the file hash from VirusTotal, the destination IP from threat intel feeds.
  2. Score — combine signals into a severity. If the hash is known-malicious and the user is in finance,
  escalate.
  3. Contain — if severity is high, isolate the host via the EDR API (cuts network access except to the
  EDR), revoke the user's session in Okta, post to the on-call Slack channel.
  4. Ticket — open a ticket in Jira/ServiceNow with all the enrichment data already attached.
  5. Wait — pause the playbook for analyst input. ("Approved to wipe the host? [yes/no]")
  6. Remediate — on approval, trigger reimage via MDM, force password reset, close the ticket.

  What used to be 90 minutes of analyst clicking becomes 3 minutes of analyst review.


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


