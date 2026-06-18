A [[Prometheus]] ecosystem component that *receives firing alerts* (e.g. from Prometheus), groups them, deduplicates them, suppresses less-useful ones, and routes notifications to humans or incident systems like PagerDuty, Slack, email, or webhooks.

Prometheus decides ***whether an alert condition is true.*** Alertmanager decides what to do with that alert, once it is firing. Should this alert be grouped with other alerts? Should it page someone? Which teams owns it? Is there an active silence? Is this alert inhibited by a more important alert? Has a notification been sent recently?
- Without Alertmanager, every Prometheus server would need to directly notify people whenever an alert expression becomes true.

Prometheus: Evaluates alert rules against metrics.
Alertmanager: Manages alert delivery, grouping, silencing, inhibition, and routing.
Grafana: Often visualizes metrics and may also have its own alerting system
