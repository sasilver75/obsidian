
A **Peer to Peer** protocol that doesn't require an intermediating server for data exchange using [[UDP]].

Ideal for audio/video calling and conferencing applications, and occasionally might be appropriate for collaborative applications like document editors, especially if they need to scale to many clients.
- Note that you often need a central storage

There are effectively four steps to setting up a WebRTC connection:
1. Clients connect to a central **signaling server** to learn about their peers
2. Cleints reach out to a **STUN server** to get their public IP address and port
3. Client share this information with eachother via the **signaling server**
4. Clients establish a direct peer-to-peer connection and start sending data

This is the happy case! In reality, sometimes these connections fail and you need to have fallbacks like our TURN server.

