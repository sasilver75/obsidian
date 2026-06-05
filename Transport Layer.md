---
aliases:
  - L4
  - Layer 4
---


With [[Internet Protocol|IP Address]]es in Hand, I can send a big block of binary data to a host, in the form of packets...

But I'm missing context. I often care about what application it's destined for, or came from. 
- This is provided by [[Port]]s
I also care about the ordering of packets, and whether they were delivered.
- This isn't provided by IP natively, but it *is* provided by some transport protocols.

Common Transport Protocols
- [[Transport Control Protocol|TCP]]: ==The default==. ==Connection-oriented==, reliable delivery, guaranteed ordering, higher latency, also has flow control and congestion control. Default, reliable.
	- When you initially establish a TCP connection, you say: "I have a sequence." and you establish a sequence number for the packets you're sending, so that if they arrive out of order or get lost, both the sender/receiver understand what's missing and missing packets can be re-requested.
		- As recipient, if you receive 3 -> 5, you will wait a second for 4 before emitting 5 to your application. There are means to re-request 4 too. 
	- The primary cost is in throughput and latency. If a packet is lost, TCP needs to retransmit it before it can continue, and if your hosts are far apart, this can take a lot of time.
- [[User Datagram Protocol|UDP]]: Connectionless, no delivery guarantee, no ordering, lower latency. Machine gun.
	- In some applications, you don't need the guarantees of TCP, and you need higher performance. As a sender, you write your data, wish it a good voyage, and send it off across the ocean and hope it makes it there. In most cases it will, and in some cases it won't.
	- This turns out to be very useful for applications like video conferencing; if you drop a frame, you don't want to pause everything, go back to the other party and say "Hey, give me that frame." Instead, you drop the frame out of the view, stabilize the audio a bit, and keep going. 
	- Useful for realtime or lossy applications (Multiplayer games, audio-video conferencing, live streaming) where latency is paramount and you can handle those failure modes (missing packets, arrival out of order). If these hold, them moving to a UDP-oriented service can be very useful.
	- ==Not supported by browsers natively.== If you have browser apps, you need to figure out a TCP version to use alongside your UDP applications (eg a mobile app using your service).
		- ((This is crazy to learn! Instead, browser use [[WebRTC]], [[WebTransport]], etc.))
- [[QUIC]]: A more modern substitute for TCP, increasingly used for web traffic, has higher performance but looks similar in terms of functionality.