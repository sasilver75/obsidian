
A rollback strategy is a predefined plane for returning a system to a previous known-good state if a change causes problems.

- Deploy rollback: Switch from version v42 back to v41
- Config rollback: Restore the previous [[Load Balancing|Load Balancer]]/[[Domain Name Service|DNS]]/firewall config
- Database rollback: Undo or compensate for a schema/data change
- Feature rollback: Disables a [[Feature Flag]]

A good rollback strategy usually defines
- What failure signals trigger rollback
- Who can approve or execute it
- What exact steps restore the previous state
- How long rollback should tak
- What data changes may not be reversible
- How to verify that the system is healthy again

==Rollback is easiest when changes are *designed to be reversible.*==
- A feature flag can be turned off quickly
- A destructive database migration might require a more careful recovery or forward-fix