---
aliases:
  - BGP
  - eBGP
  - iBGP
---
The internet's main interdomain, decentralized reachability and routing protocol. It lets independently-operated networks (Autonomous Systems) advertise IP prefixes (which [[Internet Protocol|IP Address]] ranges they can reach), and choose policy-based paths to those ranges, and eventually update routing when paths appear or disappear.

The internet isn't one network; it's a network of networks. Each large network is usually an ==Autonomous System (AS)== under on administrative control, such as an [[Internet Service Provider|ISP]], cloud provider, university, company, or [[Content Delivery Network|CDN]].

BGP solves the question:
> "Given millions of IP addresses spread across thousands of independently managed networks, how does one network learn where to send packets for a destination prefix?"

For example: If your home ISP needs to send traffic to `8.8.8.8`, the ISP needs to know which neighboring network can eventually reach Google's IP prefix. BGP is how this reachability information spreads!

Mental model:
> BGP is like networks announcing: "I can deliver traffic to this block of IP addresses," and other networks deciding whether to believe/prefer/pass along that announcement.

Note that BGP isn't just a directory, it's also a policy system; Networks don't always choose the shortest path, they often choose paths based on business relationships, cost, traffic engineering, reliability, or filtering rules.


# How BGP Works

BGP [[Routing|Router]]s form session with other BGP routers, usually over [[Transport Control Protocol|TCP]] port 179. Once a session is established, the routers exchange route information.

The central object in BGP is an IP prefix, such as:
```
203.0.113.0/24

Meaning: 203.0.113.0 ... 203.0.113.255
```

A BGP announcement say, roughly:
```
Prefix: 203.0.113.0/24
Next hop: this neighboring router
AS path: 64500 64496
Attributes: local preference, MED, communities, origin, etc.
```
Prefix: The block of IP addresses being advertised
Next hop: The router to send traffic to next
AS path: The sequence of Autonomous Systems an advertisement has passed through
Attributes: Metadata  used for routing policy and path selection


BGP is called a ==path-vector routing protocol,== meaning it doesn't advertise a complete map of the network; instead, it advertises reachable prefixes plus the AS path used to reach those prefixes.

```
AS 65010 announces: I can reach 198.51.100.0/24.
AS 65020 hears that and announces: I can reach 198.51.100.0/24 via AS 65010.
AS 65030 hears that and sees AS path: 65020 65010.
```
The AS path helps prevent loops. If an Autonomous System sees its own AS number already inside the AS path, the Autonomous System rejects that route.

==Note that BGP is Policy-Based, not Shortest-Path==
- Protocols like [[Open Shortest Path First|OSPF]] and [[Intermediate System to Intermediate System|IS-IS]] usually try to find the shortest or lowest-cost path inside one organization's network.
- BGP instead chooses routes using policy:
	- Choosing customer routes over peer routes, because customer traffic generates revenue.
	- Choosing peer routes over provider routes, because provider transit costs money.
	- Choosing one upstream provider over another because of contractual terms.
	- Choosing a longer AS path if that path is cheaper or operationally preferred.
So it's NOT that "BGP finds the best path," it's more "==BGP selects the best path according to configured route-selection rules and policy attributes, not necessarily according to physical distance, latency, hop count, or bandwidth==."

If a company's link to Provider A fails, the company withdraws the route from Provider A. Eventually, other networks stop using the Provider A path and shift to Provider B. This "eventually" actually matters: ==BGP convergence can take time.==


A BGP UPDATE message can communicate only three broad things directly: withdrawn reachability, newly reachable destinations, and path attributes describing those destinations. But the “path attributes” bucket is extensible and contains much more than AS_PATH, NEXT_HOP, ORIGIN, and LOCAL_PREF.

# Failure Modes
- BGP is both powerful and dangerous because it's decentralized. Failure modes include:
	1. Route leak: A network accidentally advertises routes it should not propagate.
	2. Prefix hijack: A network falsely advertises someone else's IP prefix.
	3. Blackholing: Traffic is routed somewhere that drops it.
	4. Route flap: A route repeatedly appears and disappears.
	5. Slow convergence: Networks take time to settle on new routes after a change.

A classic BGP problem is accidental or malicious prefix hijacking, where AS `X` announces it can reach a prefix owned by AS `Y`, and some networks believe AS `X` and send traffic there.
- Modern mitigations include Resource Public Key Infrastructure (RPKI), which lets prefix owners publish signed records saying which ASs are authorized to originate specific prefixes. This helps, but doens't magically secure all of BGP.

# Common Confusion
- BGP does not carry ordinary web traffic, it exchanges routing information. The actual user packets are forwarded by routers using forwarding tables derived partly from BGP routes.
- BGP is not DNS. DNS maps `example.com` to an IP address. BGP tells *networks* how to then *reach* those IP addresses.
- BGP is not TCP/IP itself. BGP runs *over* TCP, and BGP controls routing information for IP networks.
- BGP is not usually an internal shortest-path protocol. Inside a network, you might use [[Open Shortest Path First|OSPF]] or [[Intermediate System to Intermediate System|IS-IS]] to handle internal topology. BGP handles *policy-heavy* routing between networks and large-scale route distribution.




# External BGP (eBGP) and Internal BGP (iBGP)
- External BGP (eBGP)
	- BGP between different Autonomous Systems, used for internet routing between organizations
	- What most people mean when they say "BGP runs the internet."
- Internal BGP (iBGP)
	- BGP inside one Autonomous System, used for distributing externally learned routes within a large network
	- Used because a large ISP may have many routers, and those routers need a consistent view of external routes.














______________

A routing protocol used by routers to tell each other which IP prefixes they can reach, and what path traffic should take to get there. Routers use these advertisements to choose where to forward traffic.

In short: `BGP is how networks announce reachability for IP address ranges`

Example:
- Router A: I can reach `203.0.113.0/24`
- Router B: I can reach `198.51.100.0/24`

Other routers uses these announcements to decide where to send packets.

On the public internet, BGP is the major protocol that actually connects different networks, such as ISPs, cloud providers, CDNs, and large companies.




![[Pasted image 20260611163307.png]]
Large companies like internet service providers use this to exchange routing information with eachother.



