---
aliases:
  - SAML
---
It is an XML-based standard/protocol used for [[Single Sign-On]] (SSO), especially in enterprise systems.
- Lets a user authenticate with an identity provider (e.g. [[Auth0|Okta]], Google Workspace, OneLogin), and then access another application without logging in again.

Widely used for workplace login integrations. It is older (more painful) and more enterprise focused than either [[OAuth]] or [[OpenID Connect]], but it remains common for B2B SaaS authentication.

Flow:
1. User tries to access an app (the **Service Provider**)
2. The app redirects the user to an **Identity Provider**
3. The Identity Provider verifies the user.
4. The Identity Provider sends a signed **SAML assertion** back to the app
5. The app trusts that assertion and logs the user in.


