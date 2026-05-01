2022

Keeps [[HTTP 2]]'s semantics: same methods, status codes, headers, multiplexed streams ... but replaces the entire transport stack.
- Instead of running over [[Transport Control Protocol|TCP]] + [[Transport Layer Security|TLS]], it runs over [[QUIC]], a new [[User Datagram Protocol|UDP]]-based transport with [[Transport Layer Security|TLS]] 1.3 baked in!


### Core Motivation:
- [[HTTP 2]] fixed [[HTTP 1.1]]'s HTTP-level [[Head-of-Line Blocking]], but left [[Transport Control Protocol|TCP]]'s Head-of-Line blocking on the table!
	- Fixing TCP is essentially impossible (kernels everywhere, decades of deployment), so Google built a new transport on top of [[User Datagram Protocol|UDP]], because UDP is the only thing besides TCP that the internet's "middleboxes" (any networking devices sitting between client and server) reliably let through, and then refined HTTP on top of it.

```
HTTP/1.1, HTTP/2:   HTTP  →  TLS  →  TCP  →  IP
HTTP/3:             HTTP  →  (QUIC: TLS 1.3 + transport)  →  UDP  →  IP
```

[[QUIC]] isn't a TCP replacement bolted onto UDP as an afterthought, it's a from-scratch transport that uses [[User Datagram Protocol|UDP]] only as a delivery mechanism to traverse the existing internet. All the reliability, ordering, congestion control, and encryption logic lives in QUIC.

### What HTTP/3 itself adds (in the application layer)
- It largely reuses [[HTTP 2]]'s semantics, but rewrites the framing for QUIC streams.
	- Each request/response uses one bidirectional QUIC stream, instead of HTTP 2's stream IDs multiplexed over a single byte stream.
	- [[QPACK]] replaces [[HPACK]] for header compression. HPACK assumed in-order delivery, but QUIC delivers streams out of order, so QPACK was redesigned to allow header decoding without strict ordering, using a separate encoder/decode stream pair.
	- Frame types are similar to HTTP/2 (`HEADERS, DATA, SETTINGS, GOAWAY`), but adapted for QUIC's streaming model.

### What [[QUIC]] gives HTTP 3:
- ==Independent streams== (no transport-level [[Head-of-Line Blocking]])
	- In [[Transport Control Protocol|TCP]], one dropped packet stalls every byte after it until it's retransmitted, because TCP guarantees the in-order delivery of one byte stream. This means that with HTTP 2's multiplexing, a single lost packet stalls every stream on the connection.
	- ==QUIC's streams are independent. A lost packet on a stream A doesn't stall stream B.== Each stream maintains its own ordering separately. This is the single biggest win of HTTP/3: Multiplexing finally works correctly on lossy networks.
- ==Faster connection setup==
	- TCP+TLS 1.3 needs at leaset 2 RTT from a cold start (TCP handshake + TLS handshake), or 1 RTT with TLS 1.3 fast-open in the right conditions.
	- ==QUIC combines transport and crypto into a single handshake==:
		- 1-RTT for new connections
		- 0-RTT for resumed connections; clients can send application data in the very first packet, using cached crypto state from a previous session.
- ==Connection migration==
	- TCP connections are identified by the 4-tuple (src IP, src port, dst IP, dst port). Switching from WiFi to cellular and your IP changes, so the tuple breaks and connection dies.
	- QUIC connections are instead identified by a connection ID that's independent of IP/port, so if network changes, the connection survives. ==Mobile clients especially benefit==: long-lived connections (video calls, real-time apps, ongoing downloads) keep working, even through network handoffs.
- ==Encryption is mandatory and integrated==
	- TLS 1.3 isn't a separate layer on top of QUIC, it's *part of QUIC.*
	- even transport-level metadata (packet numbers, most of the header) is encrypted...
- Userspace implementation
	- [[Transport Control Protocol|TCP]] lives in the OS kernel. Improving TCP means waiting decades for kernels to update.
	- [[QUIC]] implementations live in [[User Space]], in browsers, servers, libraries, so they iterate at software speed.
		- Downside: More CPU per byte than TCP.
			- UDP throughput on commodity Linux was historically much worse than TCP for this reason; it's gotten dramatically better, but QUIC servers still spend more CPU than HTTP/2 servers for the same throguhput; in practice this is fine.







