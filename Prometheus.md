An open-source monitoring system that collects, stores, queries, and alerts on numeric time-series metrics. It is a **metrics collection, storage, query, and alerting system** that is often paired with [[Grafana]] (a visualization platform for observability data) to create richer dashboards.
- So yes, it *does* have its own [[Time-Series Database]] that it stores these metrics in. Grafana asks prometheus for the results of specific PromQL queries, for a time range.

Repeatedly asks your applications and infrastructure: "What are your current measurements?" saves those measurements over time, and lets you ask questions like: "How many RPS is this service handling? What percentage of requests are failing? Is latency getting worse?"

Mechanically:
1. Applications expose metrics over an HTTP endpoint, commonly `/metrics`.
2. Prometheus periodically *scrapes* these endpoints.
3. Prometheus stores each metric as a time series, identified by metric name plus labels
4. Engineers query the data using PromQL, the Prometheus Query Language
5. Prometheus evaluates alert rules, and can send alerts to [[Alertmanager]].

A [[Metric]] is a numeric measurement collected over time, such as:
```
http_requests_total = 1283912
process_cpu_seconds_total = 918.4
node_memory_available_bytes = 2310443008
http_request_duration_seconds_bucket{le="0.5"} = 34822
```
Note that metrics are different from both [[Log (Monitoring)|Log]]s and [[Trace]]s.

Prometheus has several major responsibilities:
1. Scraping Metrics
2. Storing Time Series
3. Querying with PromQL
4. Alerting



