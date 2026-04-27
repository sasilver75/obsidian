---
aliases:
  - VPN
---
Creates an encrypted tunnel between your device and another network, making your traffic appear to come from the other end of the tunnel, instead of your actual location.
- The internet is public, and sometimes you need private connectivity. VPNs make geographically separated networks act like one local network, over an encrypted tunnel.

Main use cases:
1. Privacy/geo-routing (VPNs like NordVPN, ExpressVPN): Your traffic exits through the VPN provider's server, so websites see *their* IP, instead of yours.
2. Remote Access (corporate and personal VPNs): Connect your laptop to a private network (office, home lab) as if you were physically there, so you can access internal resources that aren't exposed to the public internet.


How it works:
1. Your devices establishes an encrypted connection to a VPN server
2. Your OS routes traffic through a *virtual* network interface instead of a real one
3. The VPN server decrypts it and forwards it to the destination
4. Responses come back through the same tunnel.


[[WireGuard]] is the modern, fast, simple protocol for doing this.
- [[Tailscale]] is a product built on top of WireGuard, solving the hard parts of setting up VPNs:
	- Key distribution: Tailscale acts as a [[Control Plane]], automatically exchanging WireGuard public keys between your devices.
	- [[Network Address Translation|NAT]] traversal: Uses techniques like [[NAT Holepunching|Hole Punching]] so devices behind home routers can connect directly to eachother without [[Port Forwarding]].
	- Mesh networking: Every device on your "tailnet" can talk to every other device.
	- Zero config: You install it, log in, and your devices appear on a private 100.x.x.x network.