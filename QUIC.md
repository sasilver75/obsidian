(Originally meant "Quick UDP Internet Connections," but the [[Internet Engineering Task Force|IETF]] standardized it as just "QUIC", not an acronym)

[[Transport Layer]] protocol (~replaces [[Transport Control Protocol|TCP]]). [[HTTP|HTTP/3]] *requires* [[QUIC]].

Replaces [[Transport Control Protocol|TCP]] + [[Transport Layer Security|TLS]] both at once. It's a combined transport-and-encryption replacement. It swallows:
- TCP's job: Reliability, ordering, flow control, congestion control, congestion management
- TLS's job: Handshake, encryption, authentication

>*"QUIC = TCP + TLS, rebuilt as a single protocol on top of UDP"*

QUIC merges these two separate layers/state machines that don't know about eachother. 
- The very first packet the client sends contains both transport setup AND TLS ClientHello. One round-trip (RTT) later, you have a connection that's both reliable and encrypted, and resumption can be 0-RTT!
- The fact that QUIc can encrypt the transport metadata itself preserves its ability to evolve. TCP has "ossified" because middleboxes on the internet can read TCP's plaintext headers, and they inspect and modify TCP packets, which breaks attempts to evolve the TCP protocol.


#### UDP Usage
- [[User Datagram Protocol|UDP]] is what QUIC uses underneath; it's just a thin envelope: "here's a packet, here's its destination port, no guarantees."
- ==QUIC needed *some way* to send packets across the existing internet without middleboxes mangling them, and UDP was the only option besides TCP that the internet's [[Network Address Translation|NAT]]s, firewalls, and [[Load Balancing|Load Balancer]]s reliably pass through.==
So:
- TCP is replaced (its reliability/ordering/congestion-control role moves into QUIC)
- TLS is replaced (its handshake/encryption roles move into QUIC)
- UDP is *used, not replaced*: It's the carrier carrier that gets QUIC packets through the internet.

>"*Everything TCP did and everything TLS did is now QUIC's job. UDP is just the truck QUIC rides in on.*"

Before:
```
HTTP/1.1 and HTTP/2 stack:
  HTTP                ← application protocol
    ↓
  TLS 1.2 / 1.3       ← encryption (separate layer)
    ↓
  TCP                 ← reliable transport (in the OS kernel)
    ↓
  IP                  ← network layer
```
After:
```
HTTP/3 stack:
  HTTP/3              ← application protocol
    ↓
  QUIC                ← transport + TLS 1.3 fused into one layer
    ↓
  UDP                 ← used only as a delivery mechanism
    ↓
  IP
```

