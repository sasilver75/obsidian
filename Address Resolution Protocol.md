---
aliases:
  - ARP
  - Gratuitous ARP
  - GARP
---
The [[IPv4]] protocol  that maps an [[Internet Protocol|IP Address]] to a [[Media Access Control|MAC Address]] on a [[Local Area Network]] (LAN).

Hosts use IP addresses to decide *where* traffic should go logically, but [[Ethernet]]/[[Wi-Fi]] frames need a destination MAC Address to deliver packets on the local link.

Example:
- Host A wants to send 192.168.1.20
- Host A asks: "Who has 192.168.1.20"
- Host B replies: "192.168.1.20 is at aa:bb:cc:ddd:ee:ff"
- Host A stores that mapping in its own ARP [[Cache]]: `192.168.1.20 -> aa:bb:cc:dd:ee:ff`
- Then it can send Ethernet frames to that MAC address.


# Gratuitous ARP (GARP)
- An unsolicited ARP reply sent by a device to announce its own IP-to-MAC address mapping to the entire [[Local Area Network|LAN]]
- It is considered "gratuitous" because it is sent without the device ever receiving an APR request, effectively forcing other devices to update their ARP tables automatically.

This is useful in a [[Virtual IP Address]] Failover scenario (e.g. in [[Virtual Router Redundancy Protocol]] (VRRP)) in the case of [[Active-Passive]] High-Availability setups, where the new active node usually sends a [[Address Resolution Protocol|GARP]] announcement so that switches/routers/nearby hosts update their ARP caches quickly.