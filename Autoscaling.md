
The automatic adding or removing of compute capacity based on demand.

Instead of manually starting more servers when traffic rises, the platform watches metrics and adjusts capacity for you.

Typical flow:
1. The system monitors metrics like CPU usage, memory, request rate, queue depth, or latency.
2. A rule decides when scaling is needed (e.g. average CPU > 70% for 5 minutes, add 2 app instsances)
3. New instances, containers, or workers start up
4. Traffic is routed to new capacity
5. When demand drops, extra capacity is removed.

Commonly, it's [[Horizontal Scaling]]: Add more instances, containers, pods, or workers.

Scaling is not instant; new capacity may take seconds or minutes to become useful. Bad thresholds can cause flapping: scaling up and down too often. 




