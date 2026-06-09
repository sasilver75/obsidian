---
aliases:
  - Canary Deployment
---
A release strategy where you send a small portion of real production traffic to a new version before rolling it out to everyone.
- From the "Canary in a Coal Mine" idea: A small early group reveals problems before the whole user base is exposed to a buggy release.

Example:
1. Current version serves 100% of users.
2. Deploy new version.
3. Route 1% or 5% of traffic to the new version.
4. Watch errors, latency, logs, and business metrics.
5. If healthy, increase to 25%, then 50%, then 100%
6. If unhealthy, route traffic back to the old version.



# Comparison with [[Blue-Green Deployment]]
- Blue-Green: Switch traffic from old to new, often all at once. 
	- Clean, simple cutover, fine if you don't have enough traffic for a more complicated canary to produce useful signals.
- Canary: Shift traffic gradually, usually by percentages or user cohorts.
	- More useful when you want to reduce blast radius and validate behavior under real production traffic (assuming you have it) before a full rollout.
	- ==HIDDEN COST==: For some period, you're running two versions live at once. This means that your database schema, background workers, API contracts, frontend assets, caches, and observability all need to tolerate mixed behavior
		- This sounds ==incredibly operationally hard to make happen in a large system.==

