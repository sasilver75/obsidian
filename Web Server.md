Software that accepts [[HTTP]]/[[HTTPS]] requests and sends back [[HTTP]] responses.
- [[NGINX]], [[Apache Web Server]], Caddy, IIS
Its responsibilities are usually:
- Accept HTTP/HTTPS connections
- Handle [[Transport Layer Security|TLS]] Certificates
- Serve static files
- Route requests
- Proxy dynamic requests to app servers
- Apply compression, caching, redirects, limits, and secucrity rules.



A web server may return files directly:
```
GET /logo.png → read logo.png from disk → return image
```
Or it may forward the request to an application server (e.g. [[Web Server Gateway Interface|WSGI]]/[[Asynchronous Sever Gateway Interface|ASGI]] server):
```
GET /users/42 → forward to Gunicorn/Uvicorn → return app response
```
Subtle point: [[Gunicorn]]/[[Uvicorn]] also contain HTTP server functionality, because they can accept HTTP requests (mostly for local development's sake), but in production a "web server" usually refers to the front-facing server like [[NGINX]], while Gunicorn/Uvicorn are called Application Servers or WSGI/ASGI servers.



