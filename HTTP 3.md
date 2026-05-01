2022

Keeps [[HTTP 2]]'s semantics: same methods, status codes, headers, multiplexed streams ... but replaces the entire transport stack.
- Instead of running over [[Transport Control Protocol|TCP]] + [[Transport Layer Security|TLS]], it runs over [[QUIC]], a new [[User Datagram Protocol|UDP]]-based transport with [[Transport Layer Security|TLS]] 1.3 baked in!


### Core Motivation:
- [[HTTP 2]] fixed [[HTTP 1.1]]'s HTTP-level [[Head-of-Line Blocking]], but left [[Transport Control Protocol|TCP]]'s Head-of-Line blocking on the table!
- Fixing TCP is essentially impossible (kernels everywhere, decades of deployment), so Google built a new transport on top of [[User Datagram Protocol|UDP]], because UDP is the only thing besides TCP that the internet's middleboxes () reliably let through, and then refined HTTP on top of it.





