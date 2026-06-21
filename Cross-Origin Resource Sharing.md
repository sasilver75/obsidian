---
aliases:
  - CORS
---
A browser-mechanism that decides whether JS on one origin is allowed to read responses from another origin. CORS is enforced by browsers, not by servers themselves. A backend, `curl`, or Postman can still make the request. CORS mainly protects users from malicious websites reading data from other sites, using the user's browser credentials.

# Example

```
Website page: https://shop.example
API server:   https://api.shop.example
```
- You visit `https://shopexample`, and the page loads JS in your browser.
- This JS tries to get your shopping cart: `fetch("https://api.shop.example/cart")`
- Even though both domains belong to the same company, they're different origins, because the hostnames differ (origin = scheme + hostname + port). 
	- So the browser reads this as a cross-origin request
- The browser sends the request with an `Origin` header: `https://shop.example`
- Now, `https://api.shop.example` has to decide whether that origin is allowed to read the response. If the API allows it, the API responds with: ```
```
Access-Control-Allow-Origin: https://shop.example
```
Meaning "I allow JS from https://shop.example.com to read this response."

This is important, because imagine if you're logged into your bank at `https://bank.example` and you visit a malicious site `https://evil.example`, and that site' malicious JS tries `fetch("https://bank.example/account")`. Without CORS, that malicious JS could potentially read private account data from another site. With CORS, the `https://bank.example` response must explicitly say: `Access-Control-Allow-Origin: https://evil.example`. A real bank wouldn't do that.

Q: But if that request to https://bank.example was side-effecting... it sounds like it does the request and then the response just isn't delivered from the browser to the client JS? That still seems bad?
A: Yes, you're right. CORS by itself doesn't stop side effects. If the request changes state, then the damage might already be done. This danger is called [[Cross-Site Request Forgery]] (CSRF)

See also ==CORS Preflight requests==
- A CORS preflight request is the browser first asking the server: "Before I send this cross-origin request, do you allow this origin, method, and set of request headers?"
This might look like:
```
OPTIONS /transfer HTTP/1.1
Host: bank.example
Origin: https://app.bank.example
Access-Control-Request-Method: POST
Access-Control-Request-Headers: authorization, content-type
```
and *then* after the server returns (e.g.):
```
HTTP/1.1 204 No Content
Access-Control-Allow-Origin: https://app.bank.example
Access-Control-Allow-Methods: POST
Access-Control-Allow-Headers: authorization, content-type
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 600
```
Only then does the browser send the real request:
```
POST /transfer HTTP/1.1
Host: bank.example
Origin: https://app.bank.example
Content-Type: application/json
Authorization: Bearer abc123

{"to":"alice","amount":100}
```
This can help stop the side-effecting behavior.

```
PUT, PATCH, DELETE
Content-Type: application/json
Authorization header
custom headers like X-CSRF-Token
```
but a classic HTML form-style request often does *not* trigger preflight:
```
POST /transfer
Content-Type: application/x-www-form-urlencoded
```
This is why preflight is a CORS mechanism, not a complete [[Cross-Site Request Forgery|CSRF]] solution.


# Localhost Development Example
- This is where you commonly run into it:
```
Frontend: http://localhost:5172
Backend: http://localhost:3000
```
These *look* similar, but they are different origins, because the ports are different!
- An origin is: scheme + hostname + port

So now when your frontend code does this: `fetch("http://localhost:3000/api/user")`
The browser asks:
> "This JavaScript came from `http://localhost:5173`, but it is trying to read data from `http://localhost:3000`. Is that allowed?"

The backend needs to answer with a CORS header:
`Access-Control-Allow-Origin: http://localhost:5173`

This header means:
> "I, the backend at localhost:3000, allow browser JavaScript from localhost:5173 to read this response."

If and only if the backend sends that header, the browser gives the response to your frontend code.
If the backend does not send that header, the browser blocks your frontend code from reading the response, and you see a CORS error.
- **CORS is not the backend refusing the request. CORS is the browser refusing to hand the response to frontend JavaScript.**

Recap
1. You open http://localhost:5173
2. That page runs JavaScript.
3. The JavaScript requests http://localhost:3000/api/user
4. The browser notices this is cross-origin.
5. The backend response must include Access-Control-Allow-Origin.
6. If the header allows http://localhost:5173, the browser lets JavaScript read the response.
7. If not, the browser blocks JavaScript from reading it.

