---
aliases:
  - L3
  - Layer 3
---
At this layer is [[Internet Protocol]] (IP), whose responsibility is to ==give names that are usable to nodes on the network, and allow routing.==


IPv4 Addresses: 
- 4-bytes
- 50.46.226.113
- We've been running out of IPv4 addresses
- Typically use these extenrally, because more compatible

IPv6 Addresses:
- 16-bytes, Arranged in two-byte pairs
- 2001:db8:3333:4444:5555:6666:7777:8888
- Typically used internally

IP Addresses come in two forms:
- Public IP Addresses
	- Known to the world, assigned by a central body, and [[Routing|Router]]s are aware of them.
	- 18.0.0.0/8 is the range of addresses that Apple owns
		- This notation means the first 8 bits of the address are meaningful, meaning that first byte, here meaning 18.
		- So if you see an IP address with 18.xxx, you can send it over to the router that Apple owns.
	- Usually used for your [[API Gateway]], [[Load Balancing|Load Balancer]]s, or any externally-facing components of your design, so that people in the outside world can send them information.
- Private IP Addresses
	- You can assign your nodes whatever names whatever you'd want
	- You have to know where they are, and *your* routers need to be aware of them.
	- 192.168.0.0/16
		- This probably the address of your home router.
		- This is a special range... allocated for local networks.
		- This whole point is to eliminate overlap between private addresses and public addresses. You don't want to give your toaster an IP address that's also used by google, so that when you later try to send a packet to Google, it goes to your toaster.
	- Used for your microservices, internal hosts, etc. You need to both allocate them and be aware of them! If you're going to load-balance over a pool of services, you need to keep track of which hosts exist, where they are, and which you want to use.






