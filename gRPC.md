A high-performance [[Remote Procedure Call|RPC]] framework originally built at Google not open source under the [[Cloud Native Computing Foundation|CNCF]]. It lets services call methods on each other across a network as if thery were local function calls.

### Key characteristics:
- [[Protobuf|Protocol Buffer]]s (==Protobuf==): The default ==binary serialization format==.
- [[HTTP/2]] transport: Multiplexed streams over a single connection, header compression, bidirectional streaming.
- ==Four call types:==
	- Unary: Request -> response, like normal RPC
	- Server streaming: One request -> stream of responses
	- Client streaming: Stream of requests -> one response
	- Bidirectional streaming: Both sides stream
- ==Strongly typed contracts==: The `.proto` file is hte source of truth; Clients and servers can't drift from each other without regenerating.
- Built-in features: deadlines/timeouts, cancellation, auth, interceptor middleware, load balancing, retries.

#### Comparison vs [[Representational State Transfer|REST]]+[[JSON]]
- ✅: Faster (binary encoding, HTTP/2), strongly typed, ==great for *internal service-to-service traffic*==, first-class streaming.
- ❌: Not browser-friendly without a proxy (gRPC-Web), since browsers can't speak raw HTTP/2 framing. Harder to debug, steeper learning curve.

See also: [[ConnectRPC]]


____________________________

gRPC includes a bunch of features relevant for operating microservice architectures at scale, like streaming, deadlines, client-side load balancing, and more. The most important thing to know is that it's a binary protocol that's faster and more efficient than JSON over HTTP.

It shines in a microservices architecture where services need to communicate efficiently.

Its strong typing helps capture errors at compile time, rather than at runtime, and its binary protocol is more efficient than JSON-over-HTTP; Some benchmarks show a factor of 10x throughput!
- Consider gRPC for internal service-to-service communication, especially when performance is critical or when latencies are dominated by the network, rather than the work the server is doing.

You generally won't use gRPC for public-facing APIs, especially for clients you don't control, because having a binary protocol and the tooling for working with it is less mature than simple JSON over HTTP.
- It's more common to use gRPC for internal APIs, and have your external APIs use REST so that mobile devices and browsers can easily talk to your application.