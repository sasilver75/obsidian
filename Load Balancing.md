---
aliases:
  - Load Balancer
---


In an application, we're likely to need to [[Horizontal Scaling|Horizontally Scale]] our system.

Load Balancers provide an intermediate layer that routes clients to one of the servers in its load balancing pool.

Load Balancers can operate on multiple Load Balancing Policies:
- [[Round Robin]]
- [[Consistent Hashing]]
	- IP Address vs User ID, for example; sometimes we want to route to the same server to take advantage of cached data.


You can have **Hardware Load Balancer** or **Software Load Balancers**

Commonly, we talk about **Layer 4 Load Balancers** and **Layer 7 Load Balancers**.
- TODO: Performance tradeoffs


Load Balancers shouldn't be single points of failure, they also need to be replicated!
- **Active-Active** configuration:
	- Multiple LBs running at the same time, clients randomly choose one.
	- Load balancers can register themselves with a [[Coordination Service]] so that clients know they exist.
- **Active-Passive** configuration
	- One active load balancer, the others doing nothing.
	- Other load balancers listen to coordination service to see whether active load balancer is alive, and if not, try to claim the active role for themselves.


