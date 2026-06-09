A [[Domain Name Service|DNS]] technique where the [[Authoritative Nameserver]] returns different DNS records based on the estimated geographic location of the requester.
- `app.example.com` might resolve to:
	- a US server [[Internet Protocol|IP]] address for users in the United States
	- an EU server IP for users in Germany
	- an Asia-Pacific server IP for users in Japan

It usually estimates location from the [[Recursive Resolver]]'s IP address.
- The DNS provider (company that hosts the domain's authoritative DNS records) looks up that resolver IP in a ==GeoIP database==.
- It then applies a routing rule, such as:
	- Resolver appears to be in California -> return US-West IP

Commonly used to send users to a nearby data center, regional CDN endpoint, or jurisdiction-specific service.


The limitation is that the authoritative DNS server often only sees the DNS recursive resolver's IP. If a user in Paris uses a DNS resolver located in London, the DNS provider may think the user is in the UK.
- Some DNS systems improve this with EDNS Client Subnet, where the resolver includes a truncated prefix of the user's IP address, such as "this query is for a client in 203.0.113.0/24", then the authoritative server can geolocate the user more accurately. Not all recursive resolvers send it.


