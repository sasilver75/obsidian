---
aliases:
  - RMF
---
Defined by NIST SP 800-37, its' a structured seven-step process integrating security, privacy, and supply chain risk decisions into the system development life cycle.

Steps:
1. Prepare: Carry out the essential activities to set up and prioritize security and privacy goals at both the organization and system levels.
2. Categorize: Group the system and the information it handles based on an impact analysis of potential loss.
3. Select: Choose and tailor the correct security tools to reduce risk to acceptable levels.
4. Implement: Put the chosen security controls into action and document how they are deployed.
5. Assess: Test and verify that the controls are in place and operating correctly.
6. Authorize: Have a senior official make a risk-based choice to officially authorize the system to operate.
	- Something like an [[Authorization to Operate|ATO]] is normally the result of this step.
7. Monitor: Track system changes, ongoing control performance, and emerging threats continuously over time.




```
System definition and risk context
              │
              ▼
 Prepare → Categorize → Select → Implement → Assess
                                              │
                                  Security evidence and findings
                                              │
                                              ▼
                                  Authorizing Official
                                              │
                ┌─────────────────────────────┼─────────────────────────┐
                ▼                             ▼                         ▼
              IATT                          ATO / ATO-C                 DATO
       Limited test activity        Authorized operational use     Use denied
                │                             │
                └──────────────┬──────────────┘
                               ▼
                            Monitor
                               │
                    Changes and new findings
                               │
                    Reassessment/risk decision
```

