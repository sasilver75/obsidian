---
aliases:
  - ASGI
---
See also: [[Web Server Gateway Interface|WSGI]]

The newer Python standard for connecting a web server (e.g. [[NGINX]]) to a Python web app, especially when the app needs [[Asynchronous]] behavior. ASGI servers include [[Uvicorn]] (or Hypercorn, Daphne), for instance.

While [[Web Server Gateway Interface|WSGI]] is roughly: 
```
app(environ, start_response) -> response_bytes
```
[[Asynchronous Sever Gateway Interface|ASGI]] is instead roughly:
```
async def app(scope, receive, send):
	...
```
- scope: Describes the connection
- receive: Lets the app receive incoming events, such as request body chunks, or WebSocket messages.
- send: Lets the app sending outgoing messages, such as response headers, response body chunks, or WebSocket messages.

The server might send your app an event like:
```
{"type": "http.request", "body": b"", "more_body": False}
```
And your app might send back events like
```
  {"type": "http.response.start", "status": 200, "headers": [...]}
  {"type": "http.response.body", "body": b"Hello"}
```


The important difference: ==ASGI is built around **events**m not just one synchronous request and one response!==


Typical Pipeline:
```
  Browser / client
      ↓
  Reverse proxy, e.g. Nginx
      ↓
  ASGI server, e.g. Uvicorn
      ↓
  ASGI middleware
      ↓
  Python web framework, e.g. FastAPI / Starlette / Django / Quart
      ↓
  Application code
```
- NGINX: Accepts public HTTP(S) connections and forwards app traffic
- Uvicorn: Runs Python async workers and calls your app using the ASGI interface
- ASGI: Defines the async call contract between the server and Python app.
- ASGI Middleware: Wraps the app to inspect or modify incoming/outgoing events.
- [[FastAPI]]/[[Starlette]]/Django: Turns ASGI events into routes, request objects, responses, WebSocket handlers, and framework behavior
- Application code: Implements arbitrary endpoint behavior

# Why ASGI exists:
- [[Web Server Gateway Interface|WSGI]] works well for classic request/response HTTP
- But it's awkward for things like:
	- WebSockets
	- Server-Sent Events
	- Streaming Responses
	- Long Polling
	- Async Database/API Calls
	- Background Lifespan Events
	- Many Idle Concurrent Connections

ASGI supports those directly.

TLDR:
> WSGI is for synchronous Python web apps. [[Gunicorn]] is a common WSGI server.
> ASGI is for asynchronous Python web apps, including HTTP, WebSockets, streaming, and long-lived connections. [[Uvicorn]] is a common ASGI server.




