---
aliases:
  - Subject
---

In an [[Authentication]]/[[Authorization]] context , a Principal is the actor, the ==who or what is being authenticated/authorized==. 
- In a [[JSON Web Token|JWT]], `Subject`/`sub` usually means the subject the token is about, often similar.

Examples
- A human user: `user:alice`
- A service account: `service_account:deploy-bot`
- An application/client: `app:slack-integration`
- An organization/project: `project:acme-prod`
- A device: `device:iphone-abc123`
- Sometime a combination: `user:alice acting through app:mobile-ios`

The [[Credential]] is the thing presented to prove access (e.g. password, [[Session]] cookie, [[API Key Authentication|API Key]], [[OAuth]] access token, [[Mutual TLS|mTLS]] certification), while the [[Principal]] is the actor that the credential resolves to, and the Permissions/Scopes/Roles are what the actor is allowed to do.

