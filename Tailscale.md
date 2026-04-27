A service that makes setting up a [[WireGuard]] VPN between your devices effortless.

Without Tailscale, connecting two devices over WireGuard requires manually:
1. Generating key pairs on each device
2. Copying public keys to each peer
3. Configuring IP addresses and routing rules
4. Dealing with [[Network Address Translation|NAT]] (most devices sit behind home routers and aren' directly reachable)

Tailscale handles all of this automatically: You install the client, log in with Google/Github, and your devices appear on a shared private network (100.x.x.x addresses). Every device can reach every other device directly, no central server routing traffic, no port forwarding needed.

Key technical trick is [[Network Address Translation|NAT]] traversal: Tailscale uses a coordination server to broker the initial handshake, then the devices talk directly peer-to-peer via WireGuard.

Practical example:
> You have a home server and a laptop. Without Tailscale, accessing the server remotely means setting up port forwarding, a dynamic DNS service, etc. With Tailscale, the server just gets a stable 100.x.x.x address and your laptop can SSH to it from anywhere, as if they were on the same LAN.
> 