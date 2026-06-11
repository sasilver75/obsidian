---
aliases:
  - Decryption
  - Encrypted
  - Decrypted
---


If you want to send a message across an untrusted medium (the internet, a letter, a radio signal) such that only the intended party can read it, Encryption can help you transform readable data ([[Plaintext]]) into unreadable data ([[Ciphertext]]) using an algorithm and a key. Without the key, reversing it is computationally infeasible.


See:
- [[Symmetric Key Encryption]]
	- One key, same key encrypts and decrypts.
	- The key is fed into an algorithm along with the plaintext. The algorithm scrambles the data deterministically; the same key and same input always produces the same output. To decrypt, run it in reverse with the same key.
	- ==The problem==: How do you share the key? If you're talking to someone for the first time over an untrusted network, you can't just send them the key; anyone listening would get it too. This is the [[Key Exchange|Key Exchange Problem]].
	- ==Common algorithms==: [[Advanced Encryption Standard]] (AES) is the modern standard, used everywhere ([[HTTPS]], [[WireGuard]]), [[ChaCha20]] is an alternative to AES that's faster on hardware without AES acceleration, like mobile devices.
- [[Asymmetric Key Encryption]]
	- Two linked keys. What one encrypts, the other decrypts.
	- Exploits mathematical one-way functions (factoring, discrete logarithms, elliptic curves). You publish your public key, and keep your private key secret.
	- ==The problem==: Asymmetric encryption is slow, orders of magnitude slower than symmetric. You don't use it to encrypt actual data.
	- ==Common algorithms==: [[Rivest-Shamir-Adleman|RSA]] (old but ubiquitous, slow), [[Elliptic Curve Cryptography]] (ECC) (Based on elliptic curve discrete logarithm; much smaller keys for equivalent security. [[WireGuard]] uses Curve25519), and [[Key Exchange|Diffie-Helman]] (which is not encryption, but a key exchange protocol, but based on the same asymmetric math).


In practice, ==asymmetric encryption solves the key exchange problem for symmetric encryption!==
1. Use asymmetric crypto to securely exchange a session key (a temporary symmetric key)
2. Use that session key with AES/ChaCha20 to encrypt actual data
This is what [[HTTPS]] does via [[Transport Layer Security|TLS]], and what [[WireGuard]]! You get the security of asymmetric crypto for key exchange, and the speed of symmetric crypto for bulk data.

### Encryption at Rest vs In Transit
- In Transit: Encrypting data as it moves over a network. [[Transport Layer Security|TLS]], [[WireGuard]], [[Secure Shell|SSH]]
- At Rest: Encrypting data stored on disk. If someone steals your hard drive, they get ciphertext. Typically use [[Advanced Encryption Standard|AES]] with a key derived from your password.

### Common Pitfalls
- Encrypting but not authenticating: Encryption hides content, but doesn't prove who sent it, or that it wasn't modified.
	- (Hence authenticated encryption modes like AES-GCM)
- Reusing nonces/IVs: Most symmetric ciphers need a unique number used once per message; reusing it can completely break security.
- Weak key derivation: If your key comes from a password, you need a slow hash ([[bcrypt]], [[Argon2]]) to resist brute force. Using [[SHA-256]] directly on a password is too fast!
- Rolling your own crypto: Don't. Use audited libraries like [[OpenSSL]]. Subtle implementation mistakes break mathematically sound algorithms.




