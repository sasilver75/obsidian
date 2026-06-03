---
aliases:
  - WSGI
---
A Python standard that defines how a [[Web Server]] talks to a Python web application. Frameworks like [[Flask]], [[Django]], etc. can expose a [[Web Server Gateway Interface|WSGI]] app, and WSGI servers like [[Gunicorn]], [[uWSGI]], and others know how to run it.
- WSGI is not the web server or the web application framework, it is the contract between them.
- A WSGI Server is the process that runs your Python web application and calls it using the WSGI interface. They receive a request forwarded from (e.g.) NGINX, convert that HTTP request into WSGI input (environ, start_response), and calls your Python app: `app(environ, start_response`).
	- NGINX speaks public HTTP
	- Gunicorn speaks HTTP on one side and WSGI/Python-callable protocol on the otehr
	- Flask is the WSGI application being called

At its core, a WSGI app is a callable:
```
def app(environ, start_response):
      start_response("200 OK", [("Content-Type", "text/plain")])
      return [b"Hello"]
```
Above: 
- `environ` contains request data (path, method, headers, query string, server info)
- `start_response` is used by the app to send the HTTP status and response headers.
- The return value is an iterable of bytes which becomes the response body.

In a pipeline:
```
Browser/client
↓
Reverse Proxy (e.g. NGINX, Apache, Caddy)
↓
WSGI Server (e.g. Gunicorn, uWSGI, Waitress)
↓
WSGI middleware
↓
Python Web Framework (e.g. Flask, Django)
↓
Application Code
```
Flow:
- Browser sends HTTPS request to example.com
- [[NGINX]] receives this request
- NGINX does [[TLS Termination]] and decides this is dynamic app traffic
- NGINX forwards the request to [[Gunicorn]] at 127.0.0.1:8000
- Gunicorn converts the HTTP request into WSGI environ
- Gunicorn calls the [[Web Server Gateway Interface|WSGI]] middleware chain
- The middleware logs the request and maybe checks headers/session data.
- [[Flask]] receives the request
- Flask matches `/users/42` to `user_detail(user_id=42)`
- Your application queries the DB
- Your application code returns some JSON out from a route handler
- Flask turns that JSON into a [[Web Server Gateway Interface|WSGI]] response
- Middleware may add headers or metrics
- [[Gunicorn]] turns that WSGI response into an HTTP response
- NGINX forwards the response to the browser
- Browser displays or processes the result.

Key separation:
- NGINX accepts public HTTP(S) connections and decides where each request should go
- Gunicorn keeps Python worker processes alive and invokes your app for each forwarded request
- WSGI defines the exact Python call shape between Gunicorn and the app: `app(environ, start_response) -> response_bytes`
- Middleware is wrapper code that can inspect or change the request before the app, and the response after the app.
- Flask turns the raw WSGI request into routes, request objects, and responses, templates, sessions, and framework conventions.
- Your application code implements the actual behavior for a matched route: queries data, applies rules, calls services, returns a result.