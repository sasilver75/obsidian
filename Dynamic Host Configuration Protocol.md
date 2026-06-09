---
aliases:
  - DHCP
---
The protocol that automatically gives a device the network settings it needs to join a network.
- "Who am I?"
- "What network am I on?"
- "Where is the router?"
- "Who should I ask for DNS?"

When your laptop joins Wi-Fi, DHCP is usually how it gets:
- [[Internet Protocol|IP Address]]
- [[Subnet Mask]]
- default gateway/[[Routing|Router]]
- [[Domain Name Service|DNS]] Server
- Lease time

e.g.
```
IP address:      192.168.1.25
Subnet mask:     255.255.255.0
Default gateway: 192.168.1.1
DNS server:      192.168.1.1
Lease time:      24 hours
```

Often described as ==DORA==
- Discover: Client asks for DHCP servers
- Offer: The server offers an address
- Request: Client requests that offered address
- Ack: Server confirms the lease



