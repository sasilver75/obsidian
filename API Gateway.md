SDIAH: https://www.hellointerview.com/learn/system-design/deep-dives/api-gateway 




==In the system design interview, put the API Gateway down and move on; the API gateway is largely taken for granted in a microservices application. The only mistake you can make is to spend too much time here; it's just not the important part of your design.==
- It's important to understand every component you introduce into your design, but the API Gateway is not the most interesting. There's a far greater mistake that you are making a mistake by spending **too much** time on it than not enough.
- **Use it when you have a microservices architecture and don't use it when you have a simple client-server architecture.**


Gateways handle:
- ==Validating Requests== (are they properly formed?)
- ==Application of Middleware== (might require use of external services)
	- Authenticate requests using [[JWT]] tokens.
	- [[Rate Limiting]] to prevent abuse
	- Terminating [[SSL]] connections
	- [[Logging]] and monitoring traffic
	- [[Cache]]ing of Responses
		- Full response caching, or partial caching of specific parts of responses that change infrequently.
		- Uses TTL or event-based invalidation of cache.
	- Handling [[CORS]] headers
	- Whitelist/blacklist IPs
	- Validate Request Sizes
	- Compress responses
	- Handle response [[Timeout]]s
	- Throttling Traffic
	- Versioning APIs
	- Integrating with [[Service Discovery]]
- ==Routing Request==s to downstream services
	- Based on URL paths, HTTP methods, Query parameters, Request headers
- ==Transforming Responses==
	- (e.g. translating gRPC service responses to JSON over HTTP for clients)


**Scaling an API Gateway**
- There are two main dimensions to consider: Handling **increased load** and managing **global distribution**
	- For ==**increased load==:** While API gateways are primarily known for routing and middleware, they often include load balancing capabilities, but it's important to understand the distinction:
		- ==Client-to-Gateway Load Balancing==: Typically handled by a dedicated load balancer ***in front of*** your API Gateway instances (like AWS ELB or NGINX)
		- ==Gateway-to-Service Load Balancing==: The API Gateway itself can perform load balancing across multiple instances of backend services
			- ((This doesn't seem smart to me. Those services will still need to be in their own load balancer pool so that Service A can talk to Service B appropriately, right?))
			- ((For example, in Kubernetes, each service will likely have a LoadBalancer by default, unless the configuration has been changed to some other networking model.))
		- This can typically be abstracted away during an interview. Drawing a single box to handle "API gateway and Load Balancer" is usually sufficient. You don't want to get bogged down in the details of your entry points, as they're more likely to be a distraction from the core functionality of your system.
	- For **managing ==global distribution**==:
		- An option that works well for large applications with users spreads across the globe is to deploy API Gateways closer to our users, similar to how we'd deploy a [[Content Distribution Network|CDN]].
		- This typically involves:
			- **Regional Deployment** of gateway instances
			- **DNS-based Routing**, using [[GeoDNS]] to route users to the nearest gateway
			- **Configuration Synchronization**: Ensuring routing rules and policies are consistent across regions.

Example API Gateways:
- Managed
	- [[Amazon API Gateway]]
		- Supports REST and WebSocket APIs
	- Azure API Management
	- Google API Gateway
- Open Source
	- Kong
		- Built on [[NGINX]], extensive plugin ecosystem, supports both traditional and [[Service Mesh]] deployments.
	- Tyk
		- Native support for [[GraphQL]]
	- Express Gateway
		- JS/Node.js based


Questions:
- Q: Is it reasonable to use API Gateway as an aggregator layer, similar to the [[Backend for Frontend]] (BFF) approach that does the same thing, but introduces another microservice?
	- A: Both are valid but service different purposes. API Gateway aggregation is simpler and has lower latency but less flexible. BFF offers more power for complex transformations and client-specific needs, but adds complexity. Most large-scale systems would opt for BFF, he'd think.
- Q: If I'm just making a simple hotel website for a mom-and-pop hotel, do I still include an API gateway?
	- A: You don't need microservices, a gatway, or anything else, just a client-server, probably. But you MIGHT slap a load balancer up front and have 2 redundant instances so that you can deploy without downtime. 


--------


Let's chronicle the recent history of API gateways:

![[Pasted image 20250521125639.png]]
Our application was a single monolithic server that handles every feature, and talks to a separate datatase when needed
- Dead simple to reason about!


![[Pasted image 20250521125753.png]]
Growth forced us to slice this monolith into a bunch of different microservices.
Now the Client has an issue: It has to either know the URL of every microservice and know when to call each of them, or it would route all of its requests to Microservice 1, which would then determine which Microservice needed to be called, if any.
This was a clumsly approach! If we ever wanted to change some simple routing logic, we'd need to redeploy a microservice.


![[Pasted image 20250521125943.png]]
Fastforwarding, we introduced the first-generation API gateways, which were a thin layer in front of microservices; Now Clients only need to know one endpoint, the API gateway, which determines which microservice that request should be routed to, and then route it to that particular microservice.
- Now our routing issues are solved
- But there way another problem: ==Each microservice had a bunch of repeated boilerplate: [[Authentication]], [[Rate Limiting]], [[Logging]], etc.==




![[Pasted image 20250521130003.png]]
So we put some of these ==cross-cutting concerns== into the API gateway, letting it learn to Terminate [[Transport Layer Security|TLS]], verify Auth tokens, Throttle Abusers, log metrics, cache some common responses, etc.
- Now every single team can focus on shipping pure business logic, and the client only needs a single URL; Our gateway is in the middle, which is a shared guardrail for our entire platform.
- This is where we are today, with an API gateway sitting in front on of our microservices:
	- Main Responsibility: Routing
	- Secondary Responsibility: Cross-Cutting Concerns:


Now let's take a look at what's happening in the Gateway!
![[Pasted image 20250521130153.png]]
- See that the Gateway is responsible for four main things:
	- ==Validate the Request==: Take the incoming request and validate it: Does it have the proper formatting (headers, body, etc.)
	- ==Run the Middleware==: Oftentimes this involves us making requests to external services! If we're handling Rate Limiting, we might have a Redis cluster over on the side that the Gateway makes requests to, or if we do Authn, maybe ther's an external Authn service that we make a request to
		- Be mindful: Every request is doing this, so we want it to be fast!
	- ==Route to the correct service==:
		- We could have a basic configuration that just maps paths from the API route to various services
		- ![[Pasted image 20250521130329.png]]
- ==Transform the Response==
	- If any of the responding services were using a protocol like [[gRPC]] and the Gateway -> Client were exposing a REST endpoint to the client, then our Gateway would have to turn that binarized gRPC response into valid JSON to send over the wire to the client.





