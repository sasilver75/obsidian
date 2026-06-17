A dashboard and visualization tool often used in concert with [[Prometheus]] in the context of [[Kubernetes]].
- Grafana *can* also do alerting, but it's common to use something like [[Alertmanager]]

Grafana connects ot Prometheus as a data source -- other possible data sources include [[Grafana Loki|Loki]], [[ElasticSearch]], [[PostgreSQL]], [[Amazon CloudWatch|CloudWatch]], [[InfluxDB]], Tempo, and many more.

Grafana can produce:
- Dashboards: A collection of panels
	- Variables (e.g. `service=api`) let one dashboard work across multiple services, clusters, or environments.
- Panels: One visualization, such as a graph or table
- Graphs
- Tables
- Variables
- Annotations
- Alerting
- Access Control
- Shared Views

Typical flow:
1. You application exposes metric at `/metrics`
2. [[Prometheus]] scrapes `/metrics` every N seconds
3. Prometheus stores the values as time series
4. Grafan queries Prometheus using PromQL
5. Grafan renders charts
6. Prometheus or Grafana evaluates alert conditions
7. [[Alertmanager]] or Grafana sends notifications.