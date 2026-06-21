---
aliases:
  - CSRF
---
An attack where a malicious site tricks your browser into sending an authenticated request to another site where you are already logged in, in hopes of causing some sort of action.


Example
1. Recently, you've logged in to `https://bank.com`.
2. The bank stores your login session server-side, and gives you an opaque session [[Cookie]] in your browser to send along with future requests to the bank.
3. Later, you visit `https://evilsite.com`
4. The malicious page causes your browser to send a request to the bank:
```html
<form action="https://bank.example/transfer" method="POST">
  <input name="to" value="attacker-account">
  <input name="amount" value="1000">
</form>

<script>
  document.forms[0].submit()
</script>
```
5. On this request, your browser might automatically attach your bank cookie, since the request is going to the bank's domain.
6. From the bank server's perspective, the request just looks like "A logged-in user ABC requested a transfer," even though the user didn't make that request. The bank may treat this request as legitimate, even though this request was ==forged by another site, through the user's browser.==
	- Often times the attacker cannot even *read* the bank's response because of the Same-Origin policy (see [[Cross-Origin Resource Sharing|CORS]]), but CSRF doesn't need to read the response, it just needs to cause a state-changing action.


Typically CSRF depends on "ambient/automatic credentials," things like Cookies which the browser sends automatically. If your service's API use an `Authorization: Bearer ...` header that you JavaScript manually adds, then CSRF is usually harder, because another site can't normally force the browser to add that custom `Authorization` header.


# Main Defenses
1. ==`SameSite` [[Cookie]]s==
	- When instructed in a response, Cookies can be marked so that the browser doesn't send them in many cross-site situations:
	- `Set-Cookie: session=abctoken; SameSite=Lax; Secure; HttpOnly`
	- SameSite=Lax means "Send the cookie on normal top-level navigations, but not most cross-site form posts or subresource requests." It's the modern default, and stricter versions can break user-expected flows.
2. ==[[CSRF Token]]s==
	- The real site includes a secret random value in forms or requests.
	- The attacker's site cannot know that value, so forged requests fail.
	- ```html
	<form action="/transfer" method="POST">
	  <input type="hidden" name="csrf_token" value="random-secret-value">
	  <input name="to">
	  <input name="amount">
	</form>
	  ```
	- When the form is submitted, the server checks whether the user is authenticated, if the CSRF token is present, and whether it's correct for this session or request.
	- See the [[CSRF Token]] note for more.
3. Origin or Referer checking
	- The server rejects state-changing request unless they came from an expected origin, such as `Origin: https://bank.com`.
	- Note that the Origin is usually automatically attached by the browser on cross-origin requests and on same-origin `GET`/`HEAD` requests. A malicious website cannot normally spoof `Origin`, though a non-browser client like `curl` can send any `Origin` it wants.
4. No state changes in GET requests
	- Because attackers can trigger `GET` requests easily with images/links/scripts/redirects, and other browser features, we want to be sure that we don't create endpoints like `GET /transfer?to=attacher&amount=1000`.


# Relation to [[Cross-Origin Resource Sharing|CORS]]
- CORS protects response-reading. The attacker wants to read data from another site, like "show me the user's bank balance."
- CSR protect action-making. The attacker wants to cause an action on another site, like "transfer money from the user's account."




