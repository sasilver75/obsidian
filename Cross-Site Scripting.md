---
aliases:
  - XSS
---
A web security vulnerability where an attacker injects malicious JavaScript (or other browser-executed code) into a page viewed by other users.
- This code can then run *as if it came from the trusted site*, letting the attacker steal (unsecured) [[Cookie]]s/[[Session]] tokens, read or modify page content, impersonate the user, or performa actions on their behalf.

Example:
 > A comment field stores `<script>...</script>` is later shown to other visitors without escaping it.

Browsers typically trust JavaScript that runs on a page. If code is running inside a page from a trusted site, it can often:
- Read or change page content
- Submit forms
- Trigger actions as the logged-in user
- Access non-`HttpOnly` cookies
- Read tokens stored in [[Browser Storage]] (LocalStorage/SessionStorage/IndexedDB/Cache API)
- Send data to another server
- Modify links/buttons/UIs to trick the user


# Main types of XSS Attacks
- ==Stored XSS==:
	- The payload is persisted somewhere, like a database field, CMS field, profile name, comment, support ticket, log viewer, or admin dashboard.
		- A site lets users post comments and stores them in a database. 
		- An attacker submits `<img src=x onerror="alert('XSS')>` as a comment
		- Later, the page renders comments without escaping; every user who views the comment runs the attacker's script, which here just does an `alert('XSS')`
- ==Reflected XSS==:
	- The payload comes from the current request and is immediately reflected into the response, often through a query param, form field, path segment, or error page.
		- A search page echoes the search query into the response.
		- A victim opens this link: `https://example.com/search?q=<script>alert('XSS')</script>`
		- The server returns: `<p>You searched for: <script>alert('XSS')</script></p>`, and runs the attacker's script.
		- In this case, the payload wasn't stored; it was reflected immediately from the request into the page.
- ==DOM-Based XSS==:
	- The vulnerable behavior happens in client-side JS, where data from a source like `location.hash`, `location.search`, `postMessage`, or storage is written into a dangerous sink like `innerHTML`, `outerHTML`, or `document.write`.
		- The server returns a normal page, but frontend JS unsafely writes URL data into the DOM.
		- `https://example.com/#<img src=x onerror="alert('XSS')">`
		- In the client-side code:
			- ```javascript
			  const message = location.hash.slice(1)
			  document.querySelector(#message).innerHTML=message;
			  ```
		  - The browser turns the hash value into HTML, causing the injected `onerror` handler to run. Here, the bug is in client-side JavaScript, not necessarily in the server response.G


### Why is it called "Cross-Site?"
- It's a little confusing. It doesn't mean that an attacker is literally attacking from a different website.
- It means attacker-controlled script crosses into the trusted site’s execution context. The malicious code runs with the privileges of the vulnerable site, not the attacker’s own site.
