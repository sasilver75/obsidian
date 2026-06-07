---
aliases:
  - TLS
  - TLS Handshake
---
The protocol that gives [[HTTPS]] its security. It does three main things:
1. Authenticates the server
2. Negotiates encryption keys
3. Encrypts and integrity-protects the traffic


The sequence:
1. [[Transport Control Protocol|TCP Handshake]] establishes reliable transport
2. `ClientHello`: Client sends supported TLS versions, cipher suites, SNI hostname, ALPN protocols, random value, and key-share material.
3. `ServerHello`: Server chooses TLS version/cipher and sends its key share.
4. Both sides derive handshake keys using [[Key Exchange|ECDH]] + [[HMAC-based Key Derivation Function|HKDF]]
5. Server sends encrypted handshake messages: `EncryptedExtensions`, `Certificate`, `CerificateVerify`, `Finished`
6. Client validates certificate: `hostname matches`, `chain leads to trusted CA`, `cert not expired`, `acceptable usage/policy`, `sometimes revocation/OCSP/Certificate Transparency checks`
7. Client verifies `CertificateVerify`, proving the server owns the private key for the cert.
8. Client sends `Finished`, proves it derived the same handshake secrets.
9. Both sides derive application traffic keys
10. [[HTTP]] data can flow inside encrypted [[Transport Layer Security|TLS]] records.





_______________

When you visit `https://example.com`
1. Server sends its ==certificate== (contains its [[Asymmetric Key Encryption|Public Key]] and identity, signed by a [[Certificate Authority]])
2. Your browser verifies the certificate is signed by a CA it trusts.
3. [[Key Exchange|ECDH]] key exchange; browser and server generate ephemeral key pairs, exchange public keys, derive a shared session key (never transmitted)
4. All further communications are encrypted with [[Advanced Encryption Standard|AES]] ([[Symmetric Key Encryption]] using that session key.

The ephemeral keys are thrown away after the session; even if the server's private key is later compromised, past sessions can't be decrypted.





