---
aliases:
  - mTLS
---
A version of [[Transport Layer Security|TLS]] where both the client *and the server* [[Authentication|Authenticate]] ==each other== using certificates.
- Commonly used in service-to-service authentication

Comparison with [[Transport Layer Security|TLS]]
- Normal [[Transport Layer Security|TLS]]: Client verifies server certificate
- mTLS: Client verifies server certificate ==AND== Server verifies client certificate

Example:
- `payments-service` only requests from `orders-service` if `orders-service` presents a valid client certificate.

# Implementation

Usually you need:
1. A private/internal [[Certificate Authority]] which signs certificates for services, rather than a public CA like [[Let's Encrypt]]. Its job is to issue certificates that say things like `this certificate belongs to orders-service`.
2. Issue each service its own certificate and private key representing its identity. The certificate can be shared publicly, but the private key must stay secret.
3. Configure services to trust certificates signed by your internal CA.
4. Configure services to require client certificates.
5. On request, check the client's identity from the certificate; it's not enough to know that the client has *some* valid certificate, the server usually checks which *service identity* is in the certificate (which lets us then enforce policy, like "Actually `analytics-service` cannot create payments, sorry!")
6. Rotate certificates automatically. Certificates should expire and be replaced regularly, in an automated fashion.
7. Often, proxies or sidecars can handle this for the application (so the apps do not implement mTLS directly in their own code). The application just talks locally, while the sidecars/proxies handle certificates/TLS handshakes/identity checks/encryption.


In practice, this is often handled by infrastructure:
- [[Service Mesh]]: [[Istio]], [[Linkerd]], [[HashiCorp Consul|Consul]] Connect
- [[Proxy]]: [[Envoy]] sidecars
- Identity system: SPIFFE/SPIRE
- [[Kubernetes]] cert manager / mes CA

# Comparison with [[Service Token]]s
- A service token is usually a [[Bearer Token]] credential: `Authorization: Bearer eyJ...`
- This means that whoever has the token can use it!
- This is useful, but if the token leaks, an attacker can replay it until it expires or is revoked!

mTLS is different
- The client must prove possession of a private key during the TLS handshake, and this private key is *never* sent over the network!
- mTLS is stronger for service-to-service identity, encrypting internal traffic, preventing unauthorized workloads from connecting, etc.
	- Tokens are strong for application-level authorization, scopes and permissions, user delegation, cross-service request contexts, ... "this service can perform action X on resource Y."

In short:
- [[Mutual TLS|mTLS]] says: "This connection is from `orders-service`"
- [[JSON Web Token|JWT]]/[[Service Token]] says: "this request is allowed to create a payment for `order_123`"

This is why ==mature systems often use both:==
- mTLS for workload identity/authentication at the transport layer
- JWT/OAuth/service token for request authorization at the application layer






