https://www.hellointerview.com/learn/system-design/core-concepts/networking-essentials

Essentials:
- Networking 101
- Layer 3 Protocols: IP
- Layer 4 Protocols: TCP, UDP, QUIC
- Layer 7 Protocols: HTTP, REST, gRPC, SSE, Websockets, WebRTC
- Load Balancing
- Deep Dives on Regionalization, Timeouts and Retries, Cascading Failures

--------------
Networking is usually thought of as a layered cake:
- Top: Application-level protocols like HTTP and Websockets
- Bottom: Physical items, like wires in a network
![[Pasted image 20250519125936.png]]
Above: [[OSI Model]]
- These layers build upon one another, providing features and functionalities that you might need.

We'll focus on the **[[Network Layer]]**, where protocols like IP and Infiniband live. 
- Infiniband is what's used at AI labs, but most others will use IP.

The [[Transport Layer]] provides additional functinoality on top of IP, like guaranteed ordering and reliable delivery. 

At the very top of the stack is the [[Application Layer]], where Layer 7 protocols like [[HTTP]] and [[Websockets]] exist. These are important to you as a developer, and important in navigating tradeoffs in how your system functions and what capabilities it has.

![[Pasted image 20250519130233.png]]
- Above:
	- The client starts by resolving the domain name of the website to an IP address using [[DNS]] (DNS Resolution)
	- [[TCP]] Handshake: The client initiates a TCP connection with the server using a three-way handshake:
		- client sends a SYN, server responds with SYN-ACK, client sends ACK and establishes a connection
	- HTTP Requests
	- Server Processing
	- HTTP response
	- TCP Teardown: After the data transfer is complete, the client and server close the TCP connection using a four-way handshake
		- client sends FIN packet to server to terminate connection
		- server acknowledges the FIN packet with an ACK
		- server sends a FIN packet to client to terminate its side of the connection
		- client acknowledges the server's FIN packet with an ACK.
			- ((Does the server receive this, given that it's terminated its connection?... How does this work?))

When the [[TCP]] connection is initially established, we do a TCP handshake (Syn, SynAck, Ack), then our TCP connection is established, and we can submit requests across that TCP connection/stream.
- See that hte layers are building on eachother.
	- The HTTP layer depends on the TCP connection being established
	- The TCP layer assumes that it can depend on using IP addresses to route packets through the network. 

**NOTE:** See that there's a lot of back and forth here, which incurs **latency**.
**NOTE:** See also that we have STATE established here; we have a connection, and that connection is later terminated! We'lll see later about how to manage thsi state in our design.


![[Pasted image 20250519155039.png]]


-----------

Layer 3: [[Internet Protocol]]
- Goal: Give usable **names** that are usable to Nodes on the network, and allow **routing**.
- IPv4: 4 Bytes
	- Typically used externally
- IPv6: 16 Bytes, arranged in two-byte pairs.
	- Typically used internally
- Come in two forms:
	- Private Addresses (e.g. 192.168.0.0/16)
		- Can assign your nodes whatever names you want.
		- An example of a private IP address is 192.168.0.0/16
			- This is a special range; See the first two bytes are specified above. These are allocated for local networs; the point is to eliminate overlap between public addresses and private addresses. You don't want to give you toaster an IP addresses that's also used by Google, and when you later want to send a packet to Google, it actually goes to your toaster.
		- **Used for your internal microservices, etc. You need to both ALLOCATE them and BE AWARE of them! If you're going to load balance between IP addresses, you need to keep track of which hosts exist, where they are, and which ones you want to use.**
	- Public Addresses (e.g. 18.0.0.0/8)
		- Known to the world; assigned by a central body and routers are aware of them. 
		- 18.0.0.0/8 means that he first 8 bits are important; the first 8 bits here belong to Apple, in this case. 
		- **Usually used for API Gateway, Load Balancers, and other Externally-Facing Components of your design! People in the outside world neet to be able to send informatrion!**

-----------


Layer 4: [[Transport Layer]]
- With IP in hand, we can now send big packets/binary data to hosts... but we're missing one thing: Context! We often care about which application it's coming from and destined to! This is solved with [[Port]]s. 
- I also care about the **Ordering** of packets, as well as **Guaranteed Delivery!**

- There are 3 Transport Level protocols in use:
	- [[TCP]]: What you'll be relying on for most of your communications.
		- **Connection-Oriented ("Streams")**
			- We establish a sequence, and give packets sequence numbers. As packets get out of order or get lost, the sender/recipient will know what happened, because we have the numbers to back it up.
				- "Hey, I didn't get 2!", "Okay, I just got packets 3 and 5, I should probably wait for 4."
		- **Reliable Delivery**
			- I can't guarantee delivery in an absolute sense at a network level. This won't tell me if the data was written to disk, for instance; but it's relevant from a practical standpoint.
		- **Guaranteed Ordering**
		- **Higher Latency**
			- These guarantees have costs -- primarily in throughput and latency!
			- If a packet is LOST in transmission, TCP needs to retransmit it! If your hosts are far apart, this retransmission can take a lot of time.
		- **Flow Control, Congestion Control**
	- [[UDP]]
		- **Connectionless**
		- **No Delivery Guarantee**
			- "Spray and pray"; as a sender, you send your data across the ocean and just... hope it makes it there. In most case it will, and in other cases it won't. This is really useful in some applications like video conferencing. If we were to drop a frame, we don't want to go back to the host and ask for it, because the conversation has moved on. We'll instead just drop the frame and try to smooth the audio on the recipient side. Great for real-time or lossy applications like multiplayer games, video streaming, etc.
			- Your default should be TCP; UDP is for when latency is paramount and you have some way of handling (or not needing to handle) missed frames.
		- **No Ordering**
		- **Lower Latency**
	- [[QUIC]]: A modern alternative to TCP that's starting to get more traction, but we won't really cover this yet.


Layer 7: [[Application Layer]]
- [[HTTP]]: Hypertext Transfer Protocol
	- Even RPC or remote procedure call applications will use HTTP, because it's very versatile and battle-tested over decades.
![[Pasted image 20250519132201.png|500]]
- Note the HTTP Method, HTTP Headers (a k/v dictionary that can contain anything in or outside the HTTP standard), and the Response includes the Status Code, Response Headers, and Body.
- An important idea behind HTTP: **Content Negotation**
	- When I request some data, I might also specify to the application what type of data I can receive. (Header: Accept: application/json)
	- In the response, the application can then specify the actual content type of the returned data: (Header: Content-Type: application/json; charset=utf-8), whether or not it's what's requested.
	- This makes HTTP backwards and forwards compatible; new headers can be introduced, and clients/servers can negotiate on what formats to use.
- HTTP is broader than just web pages: It can be used for building APIs, most commonly using [[REST]].
	- REST uses the HTTP methods/verbs to describe what operations we're doing (reading/writing/updating/deleting)
	- REST uses the idea of *resources* which have URLs associated with them. If we want to read UserId:1, we might use something like `GET /v1/users/1`, and be returned some sort of JSON result describing the USER. We might optionally pass a body in our PUT request if we then want to update the user.
Basically, we're organizing our API around nouns and verbs (rather than around function calls)
- So it wouldn't be appropriate to have an `/update-user` endpoint, because that's an action, not a resource.
	- ((Though in my experience I've sometimes seen a `/actions/...` for those types of operations))

-----------

In [[GraphQL]], we're going to acknowledge some of the problems with REST designs, and try to design a solution that allows us to just fetch the data that we need.
- To populate one single mobile view, we might need to make the following REST requests:
	- GET /profile/id
	- GET /status/id
	- GET /group/id1
	- GET /group/id2
	- GET /group/id5
- So that's a lot of requests, and maybe we don't even want to render the result until we load all 5, which can be a problem if one of these endpoints is down or slow!
- So we might think Okay, maybe we just stay in the REST world and combine these all into:
	- GET /all-of-the-stuff-we-need
		- Which would return all of the data we need!
- But the problem with that is that if requirements of the UI change, we also need to change the endpoint.

So GraphQL solves the problem of:
- Constantly changing requirements of data, and not wanting to change our data on the backend.
- Not overfetching or underfetching

GraphQL provides a way for the frontend to describe the shape of the data it wants to retrieve
- The backend then figures out how to get the data so that it can return it in a way that the frontend needs.

In practice, GraphQL can be very useful when:
- Your frontend is changing all the time
- You have lots of teams, and frontend teams aren't buddies with the backend teams. 

In a system design interview, **you're** the architect of the system! It's rare that you have to design a system that can support these sort of arbitrary or changing query patterns. It's more common to use REST.

-----------

Another example of a protocol is [[gRPC]] ([[Protobuf]]s + Services)
- Protobufs provide a scehma that looks like a struct; This schema provides away ot serialize objects into very efficient binary representations. In a protobuf representation, an object might be 15 bytes, whereas in JSON, it might be 40 bytes.
- gRPC builts on Protobufs to add the idea of Services:
![[Pasted image 20250519141712.png]]
This compiles into **stubs** that can be used in a wide variety of languages!
- What gRPC does is make the serialization and deserialization of our inputs much more efficient.
	- JSON blobs are human readable, but need to be parsed... gRPC doesn't have this overhead, and can be **10x as fast/10x as much throughput** as their REST equivalents.
	- Also has some features for microservices at scale:
		- Client-side load-balancing
		- Streaming
		- Rudimentary authentication
- gRPC is how Google builds services internally, but there are two problems:
	- Mostly exteral clients do not support gRPC; Web Browsers do not support gRPC natively
	- Operating with binaries between services is effective for the servers, but makes it harder for developers and other teams to be able to debug the information being passed over the wire.
- The reality is that gRPC isn't used that often.

**In System Design interviews, gRPC is mostly useful for internal services:**
![[Pasted image 20250519141905.png|500]]
This is kind of the best of both worlds, where we have high-performance interconnects where we need it, but we have a commonly used language between the user clients and our application externally.
- **Probably don't bring up gRPC in an interview unless you're really in a high-performance scenario where you might need it.**

-----------

With [[Server-Sent Event]]s (SSE), there are examples where, rather than using a request-response format, we might want to be able to send information (stock ticker updates, notifications) that we want to send to our users.
- In this case we COULD go [[Poll]] the API, making repeated requests, but the information that we get would be delayed up to our polling frequency, and there's some overhead of repeatedly opening and closing these connections.

Server Sent Events are an extension of [[HTTP]], but with one noticable difference:
- With an HTTP request, the response is consumed almost wholly... for most cases, we don't process the response until we get everything. So if we were sending a list of responses, we don't process the events until all of them arrive...
- With SSE, we include additional headers in the response, and in the response we use newlines to designate how each of the events are happening... 
	- When the first event comes, we pass it over the wire as part of the first line of the HTTP response.
	- My client receives that first line, and ==IMMEDIATELY== starts to parse the response.
	- This basically means we have a way to **==unidirectionally push notifications from server to client==**, and the nice part is that we can use the existing HTTP machinery! I don't need new infrastructure to built on top of SSE
	- The **==downside==** is that these connections are going to be severed frequently; HTTP requests are extpected to return somewhere on the order of 30s-60s; most routers/proxies will time out requests that are longer than that. In those cases, SSE clients (eg eventsource) will automatically retry, opening a new SSE connection and passing the id of the last event that it received.
- Just because it's a little kludgey doesn't mean that it dosen't work!
	- If you need to get updates to a UI for a short period of time (like if you want to know the status of a product taht might evolve over the next few seeconds, or if you have an AI application and you want to stream responses back to the user, like a language model response).
	- These allow you to have longer running requests where the server can push events down to the client.

-----------

In some instances though, we need **bidirectional communications**, where our clients need to be pushing to the server as much as the server pushes to the client! Enter [[Websockets]]!
- WARN: often candidates use these inappropriately; They're very powerful, but often require a lot of infrastructure! **It may be more appropriate to think about a polling solution or SSE-based solution before launching into a full-tilt websocket solution, especially if the functionality we're designing is kind of narrow** (like getting a bid for an item on a page).
	- But 

WebSockets work by simulating a TCP connection, but in a way that makes it accessible for browsers and other clients. Tehy're a way of exchanging binary blobs that are guaranteed to be delivered reliably.
- When you design WS in a system design interview, you define an API of messages you're sending
	- Recall they're not requests/responses, so you don't have these input/output defintiion for API

So you'll say something like
![[Pasted image 20250519143736.png]]
You shouldn't design WebSockets with the idea of having a lot of request-response functionalitiies. If you want that, you're better off using HTTP.
- Websockest in general involve a lot of state for your application.
- That connection needs to be active for the length of that websocket connection.
- I need a way of keeping a server alive for as long as the connection needs to be alive. This can really play havoc on applicatoins where you want to do deployment, or where you might have failures.
	- So in a WS setup, you want to have some sort of fallback where you can re-establish connections or move them between servers.
	- In an interview setting, you usually have something on the periphery or edge of your design that handles the websockets and exposes methods that internal services can call. You might call it a message broker or notification service, but users will connect via WS to that service, and then that service will make requests to the internal services, and those internal services can then send messages back via websockets.

----

The final protocol we'll talk about is [[WebRTC]], which runs over [[UDP]]
- It's used in cases where you have **collaborative editors** or we have real-time **audio/visual connections** betwen clients. These are unique because it's a **Peer to Peer Connection!** It makes sense if you think about it being used for video conferencing or collaborative editors.

Flow:
1. Clients connect to a signaling server, which is basically acentralized server maintaining knowlege about all of the clients who are connected to one another, and is able to exchange information.
	1. This signaling server doesn't handle video fields.
2. The client then learns about its peers; it then learns about which peers it wants to connect to.
3. The client then connects to a STUN server
	1. (Note: Most clients don't accept inbound connections -- that would be a security problem! They're clients, not servers! But if we want clients to connect to one another, we need a process to do that! This is usually through a process called [[NAT Holepunching]], which we won't get into.)
	2. In WebRTC, this STUN Server helps us both find an address we can use (so peer can connect to us) and also facilitate that holepunching.
4. Then we can share that information between our peers so that they can connect to one another.
5. Then our clients can connect with eachother and send information over that UDP connection to eachother.
6. In the rare case where they aren't able to establish this P2P connection, they can use a TURN Server, which lets clients bounce requests to one another (this isn't a desirable outcome). 

A lot of candidates use WebRTC unnecessarily; I would avoid it unless you're doing audio or video calling; in that case, it might be worth bringing it up.
- In collaborative editors, you might use P2P connections to share updates to shared document.
- You'd also use [[Conflict-Free Replicated Datatype]]s (CRDTs) in order to share the document state in a way that's amenable to a P2P connection. If that sounds confusing, ignore it! Not necessary for 95%+ of interviews.


--------

Vertical Scaling or Horizontal Scaling
- In real life, it's much better to use Vertical Scaling when you can get away with it, which is much more often than you might imagine.
- But in interviews, it's somewhat expected that you're going to have to use horizontal scaling, and deal with the problems involved in it.

Load Balancing:
- When we have 3 Servers, how do we connect to these services?
- Load balancing splits the load and facilitates high availability; If Server1 goes down, our client can connect (via the Load Balancer reverse proxy) to Server2.
- There are two ways that we can do load balancing, at the high-level:
	- ==Client-Side Load Balancing==: The client is aware of all servers they can connect to. They query some sort of registry that tells them about the existence of all servers.
		- Alternatively, can query the Servers themselves: Maybe the Client has a reference to one server, and they can ask that server about other servers in the pool. This is how [[Redis]] Cluster works.
		- Requires that clients get updates about which servers exist, and we can't expect that our clients know about the state of servers *all the time*
		- Should use Client-Side Load Balancing in scenarios where:
			- You don't have many clients (often the case in )
			- If you have many clients, but you can tolerate update delays
				- (Example of this is [[DNS]]; When we want to resolve HelloInterview.com, we get a list; we hit the first server on the list, and if they don't respond, we get the second server on the list, etc. This lets the client handle this themselves, but... if we make an update to that IP list, it may take as long as either 5 minutes or a day for all of the DNS servers to see that update and show it to clients. So it's not a good tool to use when you have services that are scaling down or up often.)
			- Client-Side Load Balancing can be powerful, but has drawbacks; Use it for internal microservices. [[gRPC]] supports it natively, which is cool. ==Generally speaking, favor a dedicated load balancer in cases where you need to work with external clients that need to "get" updates (about changes in which servers are available) relatively quickly==
	- ==External Load Balancers==
		- Can be either Hardware LBs (F1 Networks; very powerful) or Software LBs ([[HAProxy]], Apache Web Server, [[NGINX]])... or there are cloud offerings like [[AWS Elastic Load Balancer]] or [[AWS Application Load Balancer]].
		- ![[Pasted image 20250519145252.png]]
		- When Servers are spun up, they announce themselves to the Load Balancer, saying "Hey, I'm active!" The Load Balancer will then make [[Health Check]]s on the Servers, which can either be:
			- Shallow: e.g. "Can you take a TCP connection?"
			- Deep: e.g. "Can I look at the response from your HTTP request and see the Status Code or returned data that I want to see?"
		- When a Server is healthy, the Load Balancer will send traffic to it; when it's not healthy anymore, the LB will eventually stop sending traffic to it.
		- There are also ways of routing/distributing requests to load balancers; ==Load Balancing Strategies==. Some of these include:
			- **Random**
			- **Round-Robin**
			- **Least-Connections**: Looking at all servers and finding how many connections we already have open between the LB and the server, and only allocate connections to the server that has the least connections.
				- Nice if we have a new server that we're scaling up, and we want to get it working with the others.
				- Also useful for long-running or stateful connections, like WebSocket connections, which might last for an hour! In that case, we might have uneven load, if some requests take only a few minutes, and some take a long time. In those cases, we can have pretty uneven load.
		- Another thing to think about is at what ==level== they'er operating at:
			- **[[Layer 4]] Load Balancer (TCP level)**:
				- Initiates a layer 4 connection for every inbound connection it receives.
					- The Client creates a TCP connection to the LB, and the LB creates a TCP connection to the server. ==We can kind of pretend that the LB doesn't exist; it's as if the client has a connection directly with the server, and the connection that we have to the LB is just a proxy to the client.==
				- ![[Pasted image 20250519150515.png]]
				- Layer 4 LBs are ==high performance==; it doesn't need to do a lot of thinking; It doesn't need to look at packets... all it needs to do is, when it receives a new connection, create a new connection with a server, and when it receives packets, just pushes them to the relevant server.
			- **[[Layer 7]] Load Balancer (HTTP level)**:
				- e.g. [[[AWS Application Load Balancer]]. This one, instead of only accepting TCP connections, accepts HTTP requests! It's going to choose an arbitrary server based on its load balancing algorithm, that it's going to send a request to.
				- It's not as straightforward that the connection ti has with the LB is the same one it has with its client!
				- The Layer 7 LB might have only a few connections to servers, and many connections to clients.
				- Tend to be more expensive, because they need to be able to handle a full HTTP request; But if I had a single TCP connection to a client and multiple HTTP requests that occur over it, I can then distribute those requests across the servers in my load pool, which can be very effective.
				- These are not going to be appropriate in situations where we have a connection-oriented protocol like [[Websockets]]. In those cases, we'd rather use a Layer 4 load balancer.
				- But in cases where we can accept a little performance hit, we'd like to use a Layer 7 Load balancer, which provides a lot of functionality, routing options, the ability to terminate HTTPS... It's going to be the default, and the thing we'd recommend in most instances.


Now let's talk about [[Regionalization]]. 
- When we talk about Networking across the globe, we have to talk about the laws of Physics! 
- If we want to talk from NYC to London, we're going to have 80ms of Latency no matter what we're doing! This can be a challenge.
- If we're designing a service like Uber, which operates in many locations, we can simplify problems sometimes by recognizing that riders only request drivers in the city that they're in! In that case, we can take our system design and duplicate it in the relevant cities that we're conducting business! Then we get all of the benefits of a regional network; near instantaneous responses, since our system is nice and close.
	- Typically, companies will have Datacenters that are close-by; in *regions* that are in those cities.
	- As we're regionalizing services... we want to keep our data and processing as close as possible; this is called ==colocating our data==.
- We want to get our services as close to our users as possible!
	- I can move my webserver and database to London, but I'm going to run into a few problems:
		- I can't always depend on moving my data around the world. In the Reddit case, I have users in... Japan who are accessing posts that were created from people in the UK, which was commented on by people in the US. In some sense, it's like we want our data to be everywhere! So we have a couple options:
			- ==We can have the data be local== to where it's most likely to be used (e.g. UK people accessing UK posts), so we [[Partition]] our data globally.
			- ==We can [[Replicate]] our data== to other regions so that they can read it quickly. This means that there's going to be a ==lag==; things might be available in the UK before they're available in the US.
			- We can use a ==[[Content Distribution Network|CDN]]==; We dot these edge locations around various cities in the world, and these servers are edge location that users connect to... and we make a request to it. The magic of a CDN is that they can serve many of the requests immediately (if it's hot in the cache, say... the front page of Reddit. This saves additional latency!). If the data is out of cache, then the CDN makes a request to the backing origin server, and just acts as a proxy.
- Key concepts of regionalization:
	- ==**Partition our data and keep it as close to our users as possible**==
	- ==**Try to arrange it such that our processing and data are approximately colocated.==**

-----------


Another thing you'll see in your interviews is about handling failure:
- What to do when Servers go down?
- What to do when Network Links go down?

The tools in our toolbox include:
- [[Timeout]]s,
	- When clients connect to a server, we want to make sure they eventually timeout, otherwise clients will wait forever in the case of a failed server. We want to put a sensible timeout on our request to make sure that we can 
		- We want to make sure that timeouts are short enough that users get appropriate outcomes in appropriate time, but not so short that we cut off reasonably-slow requests that are actuallyhappening, you know? 
- [[Retry|Retries]]
	- Along with timeouts, we need retries! If a request fails (because it times out), say... 5 seconds later, we make a retry!
	- Naive: 5 seconds later, we make another retry; this makes for some bunching behavior.
- [[Backoff]]s,
	- Above, we employ backoff, often **exponential backoff**, where our wait time between retries increases (say 1,2,4,8 seconds). Our first retry should happen pretty quickly; we don't want our users to wait for something that was transient... but if we get two failures in a row, it's likely something substantially is wrong, which might take a bit to recover. Exponential backoff is a way of stretching out this load and giving the server a chance to recover.
	- But this doesn't handle the synchronization/bunching problem we had above. See Jitter below....
- [[Jitter]]
	- Randomness added to our backoff (e.g. "Jittered, exponential backoff"), which ensures that the synchronization problem among multiple requests doesn't happen. Successive retries get spread out, more evenly distributing requests across the timeline.
![[Pasted image 20250519153605.png|400]]


This sounds good, and generally in Interviews, the gold standard is: ==Timeouts with Retries using jittered, exponential backoff==. That's the gold standard.

-----
Now a little bit about Cascading Failures, which are more common for Senior or Principal-level interviews.

Some of the implications that you have on your sleep/life have bitten you by now, and you want to make sure that the people that come into your company also understand the effect they can have on your life.
- We don't want systems upstream or downstream of ours to be affected by our failure!

![[Pasted image 20250519153749.png]]
Say that we have this setup.
- Say the DBA wants to do a database backup, so they initiate a snapshot process, and suddenly our database is operating at 50% of its normal efficiency.
- Server B has put some timeouts on their requests to the database, and now because the DB is returning responses 50% slower, it begins to fail some of its requests... and retrying those requests, which also fail. The server tries 3 or so requests over a while, and ultimately returns a failure.
- Server A is doing the same thing! It's making requests to Server B... some of those requests are timing out, and other are failing. Either way, Server A is retrying its request to Server B, punishing Server B for suffering under the underperforming database.
	- So Server B goes down!
	- What happens next? 
		- If we think the problem is Server B, what happens when we start it up again?
			- The database is still unresponsive to Server B requests, and Server A is still hammering Server A.
		- So ==what we need== is for Server A to give Server B a break!
		- The way to implement this is usually for Server A to implement a [[Circuit Breaker]]... a typical circuit breaker will trip when a failure exceeds a certain level, and then will periodically reset itself!
			- Server A will say: "Server B obviously isn't handling many of my requests... I'm going to temporarily pause, and give it a chance to recover before I try again. From an operational perspective, this gives us a chance to notice that Server B is slow; it's not answering request quickly because of the Database, so we fix the database problem... and then when the DB problem is fixed, Server B begins responding normally, and perhaps the circuit breaker expires and Server A begins talking to Server A again.
			- Circuit Breakers build robust systems by failing, and by failing in an obvious way.
				- We also have a bunch of **alerting** so that we can diagnose and fix the problem.




