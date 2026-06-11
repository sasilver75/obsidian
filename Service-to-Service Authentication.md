---
aliases:
  - S2S Auth
---

Usually means one backend service calling another backend service inside your system.

Comparison with [[Machine-to-Machine Authentication]] (M2M Auth):
- M2M is broader; any non-human actor authenticating to another system, including services, CI pipelines, ETL jobs, SaaS integrations, devices, or scripts.
- All S2S is basically M2M, but not all M2M is S2S.



# Problem Illustration
When `orders-services` calls `payemnts-service`, the latter needs to know:
- [[Authentication]]: Is this really `orders-service` calling me?
- [[Authorization]]: Is `orders-service` allowed to perform this action?
- Context: Is it acting for itself, or is it acting on behalf of a user?
- Scope: Is this token meant for me, or was it stolen/replayed from somewhere else?

A [[Service Token]] is a credential a service presents when calling another service, analogous to a [[Access Token]] (UAT), but issued to a service identity, rather than a human user.

Might be a:
- Opaque [[Bearer Token]]
- A signed [[JSON Web Token|JWT]]
- An [[OAuth]] access token from client credentials flow
- An internal token produced by token exchange

```
GET /internal/invoices/123
Authorization: Bearer eyJhhGbci0i...
```
The receiving service validates:
- Issuer: Trusted auth service?
- Subject: Calling service identity is as expected?
- Audience: This service?
- Signature: Valid?
- Expiration: Not expired?
- Scope/Role: Is this action allowed?


### Common Patterns

#### 1) Shared API Key
- Service A sends `X-Api-Key` header to Service B
- Simple, but usually weak: long-lived, hard to rotate, coarse permissions, easy to leak

#### 2) [[OAuth]] Client Credentials
- Service A authenticates to auth server
- Auth server issues access token
- Service A calls Service B with token
- Service B validates token

This is the standard M2M OAuth pattern.

#### 3) [[JSON Web Token|JWT]] Service Token
The token contains some signed claims:
```
{
	"sub": "orders-service",
	"aud": "payments-service",
	"scope": "payments:charge",
	"exp": 1760000000
}
```
Good, because services can validate it locally using a [[JSON Web Key Set|JWKS]] public key set.


#### 4) Token Exchange
Used when a request starts with a user, then moves through services:
- User -> [[API Gateway]] -> Service A -> Service B
- Service A should usually not forward the raw user token everywhere; instead, it may exchange it for a *new token*:
```
  subject = user-123
  actor = service-a
  audience = service-b
  scope = narrower permission
```
This preserves uesr context while limiting blast radius.


#### 5) [[Mutual TLS]] (mTLS)
- Both sides present certificates:
	- Service A proves identity with client certificate
	- Service B proves identity with server certificate
- mTLS is strong for service identity at the connection layer, often used with a [[Service Mesh]]
- But mTLS usually answers: "Who connected" better than "What application action is allowed?", so systems often ==combine== [[Mutual TLS|mTLS]] for service identity/[[Authentication]], and [[JSON Web Token|JWT]]/token scopes for request-level [[Authorization]].



#### 6) Workload Identity / Service Mesh
- In [[Kubernetes|K8s]], the best pattern is often: Avoid static secrets and let the runtime give each workload an identity.
	- Service account: The identity assigned to a service/workload
	- Workload identity: The platform mechanism that provides a running workload is that identity.
	- SPIFFE/SPIRE: Standards/tools for issuing workload identities
	- Service mesh: Infrastructure layer that can handle mTLS, identity, policy, and routing
- Mental model
	- Service account: Who the service is
	- Service token: Credential the service presents
	- mTLS certificate: Transport-level proof of service identity
	- JWT claims: Signed facts about identity, audience, scope, expiration
	- Authorizastion policy: What that identity may do


