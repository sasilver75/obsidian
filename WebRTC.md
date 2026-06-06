
A **Peer to Peer** protocol that doesn't require an intermediating server for data exchange using [[User Datagram Protocol|UDP]].

Ideal for audio/video calling and conferencing applications, and occasionally might be appropriate for collaborative applications like document editors, especially if they need to scale to many clients.
- Note that you often need a central storage

There are effectively four steps to setting up a WebRTC connection:
1. Clients connect to a central **signaling server** to learn about their peers
2. Cleints reach out to a **STUN server** to get their public IP address and port
3. Client share this information with eachother via the **signaling server**
4. Clients establish a direct peer-to-peer connection and start sending data

This is the happy case! In reality, sometimes these connections fail and you need to have fallbacks like our TURN server.



_______


![[Pasted image 20260605170603.png]]

Peer to Peer over [[User Datagram Protocol|UDP]], and used in scenarios with collaborative editors or audio-video connections (e.g. conferencing) between clients.
- Clients connect with eachother directly 
- Our clients first connect to a signaling server that maintains knowledge of all clients and can exchange information
- The signaling server doesn't handle the bulk of the information (video feeds); it can handle many clients.
- After the client connects to the signaling server, it learns about its peers, and decides which ones it wants to connect to.
- Then it connects to a STUN server. Most clients don't accept inbound connections (usually it's a security problem to do so). If we want clients to connect to one another, we need  way for them to do that.
- This is typically donee through a process called [[NAT Holepunching]].... the Stun Sever helps us to find an address we can use so that our peer can connect to us, and also facilitate that holepunching.
- Once we have an address and we're ready to go, we can share information between our peers so that they can connect to eachtother. Once they've done that, they can send information between eachother via UDP and do their video conferencing.
- In the rare case they can't establish a P2P connection, they use another centralized TURN sever, which allows clients to bounce request to eachother, which isn't an efficient or desirable outcome.

It sounds seductive, and it can be right for the right application, but many interview candidates use it unnecessarily in system design interviews.

The other use case that's acceptable is collaborative editors: You might use a P2P connection to exchange updates on a shard document, where you might use [[Conflict-Free Replicated Datatype|CRDT]]s... to share the document state in a way that's amenable to that P2P connection. 

==Not necessary for 95%+ of interviews==
