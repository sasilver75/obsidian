---
aliases:
  - CIDR
---

The modern way of describe and route [[Internet Protocol|IP Address]] ranges using an address plus a prefix length, typically used to describe the size/range of a [[Subnet]].

Aside on IPv4 address representation:
- An [[IPv4]] address is usually 32 bits, shown as `.`-delimited bytes: 0-255.0-255.0-255.0-255
- So 192.0.2.0 = 11000000.00000000.00000010.00000000

With CIDR, we can describe an IP address range, using the `address/prefix-length` form.
```
192.0.2.0/24
```
- The first N bits identify the network, and the remaining bits identify addresses inside that network.
- This represents `192.0.2.0` to `192.0.2.255`

It doesn't need to be something cleanly divisible by 8 (byte), though:
```
203.0.113.0/26
```
- IPv4 has 32 total bits, and this says we have a 26-bit prefix, so we have 32-26=6 bits remaining.
	- This then describes a range of 2^6 = 64 addresses, `203.0.113.0` through `203.0.113.63`!
- So a larger prefix equals a smaller network.


Routers store routes like:
```
0.0.0.0/0          default route
10.0.0.0/8         private network
192.168.1.0/24     local subnet
192.168.1.128/25   more specific subnet
```
When a router sees a destination IP address, it uses longest prefix match: Choose the matching route with the most fixed bits. 
- So for `192.168.1.140`, the router would choose `192.168.1.128/25` over `192.168.1.0/24` if both were available.



# Comparison with [[Subnet Mask]]

These are two ways of expressing the same network boundary!

```
CIDR: `192.168.1.0/24`
Subnet Mask: `192.168.1.0` with subnet mask `255.255.255.0`
```


# What about IPv6?
- It can use CIDR-style prefix notation too!
```
2001:db8:abcd::/48
```
IPv6 addresses have 128 bits rather than 32, so /48 means that the first 48 bits are prefix, and the 80 remaining bits describe the size of the network.




