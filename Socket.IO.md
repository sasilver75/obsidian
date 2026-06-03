
A [[Real-Time Communication]] library that lets a server and client send events to eachother with low latency, typically building on top of [[WebSockets]]. Use when you want a richer real-time app layer that handles common difficulties for you.
- Client: JavaScript in the browser, via `socket.io-client`
- Server: Typically Node.js via the `socket.io` npm package, but there are compatible clients in Python/Java/Swift/C__/Go/Rust/.NET.

Commonly used for:
- Chat and notifications
- Multiplayer or collaborative applications
- Live dashboards
- Presence indicators
- Streaming state updates, like map/unit/inventory changes

It USUALLY uses [[WebSockets]] underneath, but ==it is not the same as raw WebSocket!== 
- It ==adds its own protocol and features==, including fallback transports like [[Long Polling]], automatic reconnection, event acknowledgements, packet buffering, broadcasting, rooms/namespaces, and multi-server scaling support.


```javascript
// server
io.emit("inventory:update", { unitId: "325-bsb", class: "III" });

// client
socket.on("inventory:update", (update) => {
  console.log(update);
});
```



