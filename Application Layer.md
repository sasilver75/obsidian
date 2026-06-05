---
aliases:
  - L7
  - Layer 7
---


[[HTTP|HyperText Transfer Protocol]] (HTTP), the most popular protocol for a lot of applications, not just websites. Even [[Remote Procedure Call|RPC]] applications will use HTTP because it's very versatile and battle-tested over decades.

Works by simple text-formatted requests and responses:
![[Pasted image 20260605134101.png]]
- GET /posts/1  with a version of HTTP ([[HTTP 1.1]]), along side with some headers (a KV dict that can contain anything, including some widely-accepted ones like content-encoding from the standard). The method is called an HTTP verb, which also tells intent.
- The response includes the actual data we want to transmit, but also the status code, which tells us whether it was successful or not, as well as some headers, which might indicate what format the data is in, or the date that it was retrieved (e.g.).

One of the more important ideas behind HTTP is [[Content Negotiation]]
- When making a request, I might tell the web server I'm request it from what kind of things I can receive.
- I can accept compressed content.