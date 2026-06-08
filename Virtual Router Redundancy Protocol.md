---
aliases:
  - VRRP
---
A network protocol that lets multiple [[Routing|Router]]s or [[Load Balancing|Load Balancer]]s share a single [[Virtual IP Address]] so that one device can fail and another can take over without clients changing their gateway/service address.


A VRRP group usually has:
- A virtual IP (e.g. 203.0.113.10)
- A master, which currently owns and answers for that VIP
- One or more backups, which monitor the master
- A priority, used to decide who should become master
- Periodic advertisements, sent by the master to prove it is alive

# Example:

```
VRRP group 42
VIP: 203.0.113.10
LB-A priority 150  -> master
LB-B priority 100  -> backup
```

  - LB-A periodically sends VRRP advertisements saying, effectively: "I am alive. I am still master for VIP 203.0.113.10."
  - LB-B listens. As long as those advertisements continue, LB-B stays passive.
  - If LB-B stops hearing them, it assumes LB-A is dead or unreachable and promotes itself to master.

The Actual Failover mechanism:
- The important part is not just “LB-B becomes active.” The network also has to learn that the VIP now lives at LB-B. 
- On an Ethernet/[[Local Area Network|LAN]] network, devices use [[Address Resolution Protocol]] (ARP) for IPv4:
	- Who has 203.0.113.10?
	- Tell me the MAC address.
- Before failover, the answer is LB-A’s MAC address.
- After failover, LB-B must announce:
	- 203.0.113.10 is now at LB-B's MAC address.
- It typically does that using gratuitous ARP.
- For IPv6, the equivalent is handled through Neighbor Discovery, using unsolicited Neighbor Advertisements.

So failover looks like this:
1. LB-A owns VIP 203.0.113.10.
2. LB-A sends VRRP advertisements.
3. LB-A fails.
4. LB-B stops receiving advertisements.
5. LB-B elects itself master.
6. LB-B adds 203.0.113.10 to its interface.
7. LB-B sends gratuitous ARP.
8. Switches, routers, and hosts update their ARP tables.
9. New traffic to 203.0.113.10 goes to LB-B.

That is the practical mechanism behind [[Active-Passive]] [[Virtual IP Address|VIP]] failover.