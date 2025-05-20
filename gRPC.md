


gRPC includes a bunch of features relevant for operating microservice architectures at scale, like streaming, deadlines, client-side load balancing, and more. The most important thing to know is that it's a binary protocol that's faster and more efficient than JSON over HTTP.

It shines in a microservices architecture where services need to communicate efficiently.

Its strong typing helps capture errors at compile time, rather than at runtime, and its binary protocol is more efficient than JSON-over-HTTP; Some benchmarks show a factor of 10x thoughput!
- Consider gRPC for internal service-to-service communication, especially when performance is critical or when latencies are dominated by the network, rather than the work the server is doing.

You generally won't use gRPC for public-facing APIs, especially for clients you don't control, because having a binary protocol and the tooling for working with it is less mature than simple JSON over HTTP.
- It's more common to use gRPC for internal APIs, and have your external APIs use REST so that mobile devices and browsers can easily talk to your application.