---
aliases:
  - OSI Model
  - OSI
---
Open Systems Interconnection (OSI) Model, a conceptual model created by [[International Organization for Standardization]] (ISO) which enables diverse communication systems to communicate using standard protocols.


1. Physical Layer: Transmits raw bits over a physical medium: electrical signals, radio waves, light pulses, cables, connectors, frequencies.
2. Data Link Layer: Moves frames across a single local network link. Handles local addressing and media access. Examples: [[Ethernet]], [[Wi-Fi|WiFi]] [[Message Authentication Code|MAC]] layer.
3. Network Layer: Moves packets between networks using logical addressing and routing. Examples: [[Internet Protocol]]
4. Transport Layer: End-to-end communication between processes, including reliability, ordering, flow control, and ports. Examples: [[Transport Control Protocol|TCP]], [[User Datagram Protocol|UDP]], [[QUIC]]
5. Session Layer: Manages conversations between systems: starting, maintaining, and ending sessions. Often blurred into application protocols, today.
6. Presentation Layer: Data format, encoding, compression, and encryption concerns. Real systems don't always map cleanly here, but sometimes [[Transport Layer Security|TLS]]-style encryption is put here.
7. Application Layer: User-facing network services, such as web browsing, email, file transfer, and domain name lookup. Examples: [[HTTP]], [[Domain Name Service|DNS]], [[Simple Mail Transfer Protocol|SMTP]]



![[Pasted image 20260423213546.png]]

These layers help eachother, and area also oblivious some of the implementation details of eachother.