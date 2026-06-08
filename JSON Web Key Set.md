---
aliases:
  - JWKS
  - JWK
  - JSON Web Key
---
A published set of public keys that services use to verify [[JSON Web Token|JWT]] [[Cryptographic Signature|Signature]]s.

- Authentication service signs a new [[JSON Web Token|JWT]] with a [[Asymmetric Key Encryption|Private Key]]
	- The auth service exposes something like `https://auth.example.com/.well-known/jwks.json`, which other servers can retrieve when needed and store in their locak JWKS cache.
- Other services then verify incoming [[JSON Web Token|JWT]]s with the public key from [[JSON Web Key Set|JWKS]]
	- When a service receives an incoming JWT:
		1. Decode JWT header
		2. Read `kid = key-123`
		3. Fetch/cache [[JSON Web Key Set|JWKS]] from trusted issuer
		4. Find public key with `kid=key-123`
		5. Verify signature
		6. Validate claims: `iss`, `aud`, `exp`, `nbf`, scopes.
		7. Authorize the request

The JWKS contains public keys, with fields like:
```
  {
    "keys": [
      {
        "kid": "key-123",
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "n": "...",
        "e": "..."
      }
    ]
  }
```



__________

A JSON document containing the set of [[Asymmetric Key Encryption|Public Key]]s that a server uses to sign [[JSON Web Token|JWT]]s.
- Clients (or resource servers) fetch it to *verify token signatures* without having to hard-code or pre-share keys!
- Enables [[Key Rotation]]: the issuer can publish a new key, start signing with it, and old/new keys can coexist in the set during the transition.
- Verifiers don't need shared secrets, just network access to the JWKS URL (and cache the result, typically)
- Standard format (RFC 7517) so any compliant JWT library can consume it.

# JWT Flow
1. The issuer (e.g. [[Auth0]], [[Supabase]], etc.) signs [[JSON Web Token|JWT]]s with a [[Asymmetric Key Encryption|Private Key]]
2. It publishes the matching public keys at some well-known JWKS endpoint (e.g. `https://issuer.example.com/well-known/jwks.json`)
3. Each JWT's header includes a ==kid== (key ID)
4. The verifier fetches the JWKS, finds the key whose `kid` matches, and uses it to verify the signature.


Shape of a JWKS:
```json
{
  "keys": [
    {
      "kty": "RSA",
      "kid": "abc123",
      "use": "sig",
      "alg": "RS256",
      "n": "...",   // modulus
      "e": "AQAB"   // exponent
    }
  ]
}
```

