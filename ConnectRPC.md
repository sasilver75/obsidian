---
aliases:
  - Connect
---
An RPC framework built by Buf that's designed as a ==more ergonomic, browser-friendly alternative to [[gRPC]], while staying interoperable with it.==

Key points:
- ==Same `.proto` files==: Define services in protobuf, exactly like gRPC. Reuses the protobuf ecosystem (codegen, types, tooling)
- ==Three protocols on one server==. A Connect server simultaneously speaks:
	- Connect protocol: Buf's own simpler HTTP/1.1 and [[HTTP 2]]-friendly protocol
	- gRPC: Full compatibility with existing gRPC clients
	- gRPC-Web: For browsers without a proxy
- ==Browser-native==: Connect-Web works directly from the browser over HTTP/1.1 or HTTP/2, no Envoy/grpc-web proxy required. ==This is the big draw vs raw gRPC==.
- Smaller, simpler runtime: The Go and TS implementations are much lighter than the official gRPC libraries.
- Languages: Supports Go, ES, Kotlin ,Swift, Python

ConnectRPC exists because raw gRPC is painful from browsers (needs gRPC-Web and a proxy like Envoy), debugging it is hard (binary-only), and the official libraries are heavy.
- Connect keeps the protobuf contract and the gRPC wire format compatibility, but also adds a JSON-over-HTTP option and trims the runtime.

