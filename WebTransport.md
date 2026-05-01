A modern, low-latency protocol designed to address the architectural limitations of WebSockets by leveraging [[HTTP 3]] and [[QUIC]].
- [[WebSockets|WebSocket]]s are still industry standard for general real-time use, but WebTransport is superior for high-performance applications like cloud gaming and live streaming.

> "WebSockets, but built on [[QUIC]], with both reliable streams and unreliable datagrams."
- Runs over [[HTTP 3]], which itself runs over [[QUIC]] ([[User Datagram Protocol|UDP]]-based, with [[Transport Layer Security|TLS]] 1.3 baked in). No [[Transport Control Protocol|TCP]] head-of-line blocking.
- A single WebTransport session multiplexes:
	- Reliable, ordered streams (like [[Transport Control Protocol|TCP]]/[[WebSockets|WebSocket]]), either bidirectional or unidirectional.
	- Unreliable datagrams: Fire-and-forget [[User Datagram Protocol|UDP]]-style packets. Lost ones aren't retransmitted.
	- Connection setup is fast (thanks to QUIC's 0-RTT or 1-RTT handshake) and survives network changes (switching Wi-Fi to cellular without dropping)

Browser API:
```javascript
const transport = new
WebTransport('https://example.com:4433/wt');
await transport.ready;

// Reliable stream
const stream = await
transport.createBidirectionalStream();
const writer = stream.writable.getWriter();
writer.write(new TextEncoder().encode('hello'));

// Unreliable datagram
const dgWriter =
transport.datagrams.writable.getWriter();
dgWriter.write(new Uint8Array([1, 2, 3]));
```

Comparison with [[WebSockets|WebSocket]]
![[Pasted image 20260501113601.png]]

![[Pasted image 20260428170654.png]]

