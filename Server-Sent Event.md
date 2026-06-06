---
aliases:
  - SSE
---
A *unidirectional communicaiton protocol* that's a web standard (part of HTML5) that ==lets a server asynchronously push data to a client over a single long-lived HTTP/1.1 or HTTP/2 connection.==

Unlike traditional HTTP request/response, where the client must repeatedly poll the server for new data, SSE keeps the connection open and lets the server stream events whenever they occur. The transport is plain HTTP with a `Content-Type: text/event-stream` response header, and the body is a sequence of UTF-8 encoded text frames.

SSE is ==the simplest way to stream server-pushed updates to a browser==: it's ==just HTTP with a text/event-stream body, has built-in reconnection and event IDs, and is supported natively by every modern browser== via EventSource. Reach for it when you need one-way, text-based, real-time updates — and reach for WebSockets when you need bidirectional or binary.

How it works:
1. Client opens a connection: A normal HTTP GET with `Accept: text/event-stream`
2. Server responds withs streaming body, setting `Content-Type: text/event-stream`, `Cache-Control: no-cache`, and `Connection:keep-alive`. It does NOT close the response.
3. Server emits events, writing text frames to the body whenever it has data.
4. Client parses events; the browser's EventSource parses the stream and dispatches events to handlers.
5. Auto-reconnect: If the connection drops, the browser automatically reconnects after a delay (configurable) and sends the last received event ID via the `Last-Event-ID` header so the server can resume.

In the actual ==Wire Format==, events are newline-deliminted text; each event is a group of fields, terminated by a blank line:

```
  event: priceUpdate
  id: 42
  retry: 5000
  data: {"symbol":"AAPL","price":182.55}

  data: line one
  data: line two

  : this is a comment (heartbeat)
```
  Fields:
  - data: — the payload (multiple data: lines are concatenated with \n).
  - event: — optional event type (defaults to "message").
  - id: — optional ID; the browser sends it back as Last-Event-ID on reconnect.
  - retry: — reconnection delay in ms.
  - Lines starting with : are comments, commonly used as heartbeats to keep proxies from
   timing out the connection.

## Use Cases
- LLM token streaming: Both Anthropic and OAI stream model output via SSE; each token (or content delta) is a separate event.
- Live dashboard and monitoring
- Notificaions and Activity feeds
- Financial tickrs
- Progress updates for long-runing jobs (eg video transcoding)



## Client API (in Browser)
```javascript
Client API (browser)

  const es = new EventSource("/api/stream");

  // Default "message" events
  es.onmessage = (e) => console.log("data:", e.data);

  // Named events
  es.addEventListener("priceUpdate", (e) => {
    const tick = JSON.parse(e.data);
    console.log(tick.symbol, tick.price);
  });

  es.onerror = (e) => console.error("stream error", e);

  // Close when done
  es.close();

  Server example (Node.js)

  app.get("/api/stream", (req, res) => {
    res.set({
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    res.flushHeaders();

    const send = (data, event) => {
      if (event) res.write(`event: ${event}\n`);
      res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    const interval = setInterval(() => send({ ts: Date.now() }, "tick"), 1000);
    req.on("close", () => clearInterval(interval));
  });
```



# Comparison
- With [[WebSockets|Websocket]]
- With [[Long Polling]]
![[Pasted image 20260427174007.png]]
Generally: 
- If you need server -> client only, use SSE.
- If you need bidirectional, low-latency, or binary: use WebSockets
- For occasional updates with simple semantics, use polling.


__________

![[Pasted image 20260605164351.png]]
For stock ticket updates, you *could* poll the API to make repeated requests, but that means:
- Your information will be delayed by up to your poling frequency
- Overhead of repeatedly  opening/closing connections and making requests

SSE is an extension on HTTP with one notable difference:
- With a typical HTTP request, our response is consumed almost wholly (this isn't exactly true)... so if I were sending a list of events that happen over the course of 10s, I don't process those events until all complete and I get the response.
- With SSE, I include additional headers in the response (chunked-encoding) and in the response I use newlines to designate how each of the new events are happening. Because of the headers, the proxies/other things handling the request will send that response on to my  clients, and they'll being parsing the response.
	- So I have a way of unidirectionally pushing data from my server to a client using existing HTTP machinery.
	- Downside: These connections are going to be severed frequently! HTTP requests are expected to return in the 30s-1m mark, so many routers/proxies will disconnect requests that exist longer than that. In those cases, our SSE client will automatically retry, opening a new SSE connection, passing the ID of the last event it received, in case events happened between the loss of connection and opening of responses.
		- It's a bit of a kludge: we're existing on a connection that's constantly pushing responses which is periodically severed and re-enabled. Just because it's kludgey doesn't mean it doesn't work!
- Use Cases
	- If you're buying a product and you want to know the status that might evolve over the next 15 seconds
	- For AI applications where you might want to stream tokens or responses back to the user, which might take awhile (e.g. 30 seconds).

SSEs build on HTTP and let you have longer running requests and let the server unidirectionally push events down to the client.

If we need bidirectional communication, we would use something like [[WebSockets|WebSocket]]

