---
aliases:
  - TLS
---


When you visit `https://example.com`
1. Server sends its ==certificate== (contains its [[Asymmetric Key Encryption|Public Key]] and identity, signed by a [[Certificate Authority]])
2. Your browser verifies the certificate is signed by a CA it trusts.
3. [[Key Exchange|ECDH]] key exchange; browser and server generate ephemeral key pairs, exchange public keys, derive a shared session key (never transmitted)
4. All further communications are encrypted with [[Advanced Encryption Standard|AES]] ([[Symmetric Key Encryption]] using that session key.

The ephemeral keys are thrown awy after the session; even if the server's private key is later compromised, past sessions can't be decrypted.