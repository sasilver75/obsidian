---
aliases:
  - SSO
---
A user signs in once through a trusted identity system, and then can access multiple apps without entering credentials separately for each app. "Log In with X" is a common UI for doing thing.
- SSO is the user experience pattern: one login for many apps
	- ✅ [[Security Assertion Markdown Language|SAML]] is a common protocol for implementing SSO in enterprises
	- ✅ [[OpenID Connect|OIDC]] is one more modern protocol that can implement that pattern
	- ❌ [[[OAuth|OAuth 2.0]] itself is mainly for authorization, not authentication, though it is often used alongside OIDC.

Example: 
> You sign in to Google Workspace, then open Gmail, Drive, Slack, and a company HR app without separate logins. 
> Those apps all trust the identity provider to confirm who you are.
- Authentication: Google still verifies who you are; if you already have an active Google session in your browser, Google can immediately say "Yes, this is Alice" without asking for your password again. 
- App Session: Each app still creates its own session after Google confirms your identity. You might be redirected through Google for each app's sign on, but you won't (again) be asked for your password, because Google already knows that you're signed in.


1. You sign in to Google Workspace once.
2. You open Slack.
3. Slack redirects you to Google.
4. Google sees you already have a valid Google session.
5. Google sends you back to Slack logged in.
6. You open the HR app and the same thing happens.