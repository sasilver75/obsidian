---
aliases:
  - Machine Token
---
A credential issued to a non-human principal (e.g. backend service, batch job, deployment pipeline, or third-party integration) so that another system can authenticate and authorize that machine actor.


Example: Nightly Billing Job
- Imagine that we have a scheduled `billing-invoice-exporter` that runs every night, which needs to:
	- Read invoices from the billing API
	- Write finalized invoice PDFs to object storage
	- Notify the payments API that invoices are ready
- Instead of using a human engineer's account, the job has its own service identity:
```
svc:billing-invoice-exporter-prod
```
- At runtime, the job obtains a short-lived service token from an internal identity provider, which means: "The caller is the production billing invoice exporter service, and it is allowed to perform these specific actions for this specific audience until this expiration time."
- The bearer token can then be used by the job to calls APIs like this:
```http
GET /invoices/pending
Authorization: Bearer eyJhbGciOiJSUzI1NiIs...
```

Typically, short-lived internal service/machine access token are [[Structured Token]]s, while external or long-lived service credentials are often [[Opaque Token]]s (because of revocability benefits of storing token information server-side).

==The most modern pattern is often not to store a permanent service token at all.== Instead, the workload proves its identity using a stronger underlying mechanism, such as cloud instance identity, Kubernetes workload identity, [[Mutual TLS]], or a private key, and receives a short-lived access token.

# Structured Token Example
The decoded token payload might look like:
```json
{
  "iss": "https://auth.internal.example.com", -- Token issuer, e.g. internal authz server
  "sub": "svc:billing-invoice-exporter-prod", -- Service identity the token represents
  "aud": "billing-api", -- The intended recipient API
  "scope": "invoices:read invoices:export", -- e.g. the actions the server can perform
  "iat": 1781452800, -- When the token was issued
  "exp": 1781453700, -- When the token expires
  "jti": "tok_01JZK8V6Q3M9B4P7R2D1" -- A unique token identifier, for logging/revocation
}
```
- The receiving API can often validate the token locally by checking the issuer's signature, expiration, audience, and scopes.
- The structured token should not contain secrets, passwords, db credentials, etc. It should contain enough claims to verify the identity and authorize the caller, but not become a portable dump of sensitive state.
# Opaque Token Example
The opaque token might look like this:
```
st_prod_6Vb92nQp7YwLk3RxA0sDqF8m
```
- The token itself has no readable meaning to the receiving service; it must ask an authentication server, database, or (e.g.) token introspection endpoint:
```http
POST /introspect  %% Example endpoint; point being "someone that knows what it means" %%
token=st_prod_6Vb92nQp7YwLk3RxA0sDqF8m
```
The returned server state might be:
```json
{
  "active": true,
  "subject": "svc:billing-invoice-exporter-prod",
  "audience": "billing-api",
  "scopes": ["invoices:read", "invoices:export"],
  "issued_at": "2026-06-14T02:00:00Z",
  "expires_at": "2026-06-14T02:15:00Z",
  "created_by": "workload-identity-system",
  "last_used_at": "2026-06-14T02:03:12Z",
  "last_used_ip": "10.24.8.19",
  "revoked_at": null
}
```
Then the API can make authorization decisions based on this.

Opaque tokens are useful when you want central control: immediate revocation, server-side rotation, richer audit state, or the ability to change permissions without waiting for a structured token to expire.


Scenario                                  | Typical form                                      | Why
------------------------------------------|---------------------------------------------------|------------------------------------------------------------
Internal service-to-service call           | Structured JWT, sometimes PASETO or SPIFFE SVID   | APIs can validate locally without database lookup
OAuth 2.0 client credentials access token  | Often structured in internal systems; sometimes opaque in public APIs | Depends on whether the resource server needs stateless validation
Kubernetes service account token           | Structured JWT                                    | Kubernetes and related systems can verify issuer, audience, expiration
Cloud workload identity / OIDC federation  | Structured JWT for identity assertion             | The token proves workload identity to another system
SaaS API key for an integration            | Opaque                                            | Provider wants central revocation, rotation, metering, and audit control
CI/CD secret token                         | Often opaque                                      | Simpler to revoke and track server-side
Long-lived “service token” in a dashboard  | Usually opaque                                    | Long-lived bearer credentials should not expose embedded authorization state