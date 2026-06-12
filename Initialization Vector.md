---
aliases:
  - IV
---








# Relation between [[Nonce]]s, [[Salt]]s, and [[Initialization Vector|IV]]s
All three are usually extra values added to cryptographic operations, but they're used in different places.
The terms overlap; in many modern encryption systems, people may call the same value a nonce or an IV, depending on the documentation. Partly historical distinction, partly algorithm-specific.

- [[Salt]]: Makes a stored hash unique, usually used in password hashing or key derivation.
	- "A salt makes each password-hash record its own separate target."
	- Usually public, random, stored with the hash, and unique per password or derived key.
- [[Nonce]]: Makes one cryptographic operation unique, usually used to make an operation happen only once under the same key.
	- "A one-time label that keeps repeated operations from reusing the same cryptographic setup."
	- Usually public, unique, used once per key/context. Sometimes random, sometimes a counter.
- [[Initialization Vector]] (IV): Sets the starting state for an encryption algorithm, mostly used in block cipher modes.
	- "The starting position for an encryption mode; 'what starting state should this encryption run begin with, so that identical plaintext don't encrypt in predictable ways?'"
	- Usually public, used by encryption modes, required by the specific encryption mode, sometimes required to be random, sometimes only unique.
