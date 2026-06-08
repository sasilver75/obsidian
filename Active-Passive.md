A High-Availability (HA) pattern where, though you have two or more systems available, only one is serving traffic at a time, with the rest waiting to take its place in the case of failure.

In [[Load Balancing]] terms:
- An Active node handles all live traffic
- A Passive/standby node stays ready, but not not normally receive traffic.
- If the active node fails, the passive node is promoted and starts serving traffic, in a process called Failover.

> ==Active-Passive usually works by moving a network identity, not asking the clients to change where they're connecting.==


We use a [[Virtual IP Address]] (VIP), which is an IP address that isn't permanently owned by one physical machine. Instead, it "floats" between machines.

In an active-passive load balancer pair:
```
Clients
    |
    | connect to 203.0.113.10
    v
  VIP: 203.0.113.10
    |
  Active LB-A       Passive LB-B
  owns VIP          standby
```
So Clients, DNS, upstream routers, and applications all use the same IP: 203.0.113.10
They do not know or care about whether LB-A or LB-B is currently active.

When LB-A fails, LB-B takes ownership of the same VIP. ==The service address remains stable while the physical owner changes.==

So how do we coordinate which machine currently owns the VIP?

We use [[Virtual Router Redundancy Protocol]] (VRRP), a protocol used to coordinate which machine currently owns the VIP. It's commonly used for routers, firewalls, and load balancers.

A VRRP group usually has:
- A virtual IP (e.g. 203.0.113.10)
- A master, which currently owns and answers for that VIP
- One or more backups, which monitor the master
- A priority, used to decide who should become master
- Periodic advertisements, sent by the master to prove it is alive





# Comparison with [[Active-Active]]
- In Active-Active, multiple (e.g.) load balancers handle traffic at the same time, though you need to figure out a routing/steering mechanism to deliver traffic among them.
- Active-Passive: Often used when you want simpler high availability, but it has tradeoffs:
	- Some capacity sits idle most of the time
	- Failover may take seconds or longer
	- The passive node must stay in sync with the active node if it needs current state/data
