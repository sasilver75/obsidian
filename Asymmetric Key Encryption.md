---
aliases:
  - Public Key
  - Private Key
  - Asymmetric Encryption
---


A Key pair is two mathematically linked numbers. What one key encrypts, only the other key can decrypt.

Public key: Share it freely with anyone. It's useless without the private key.
Private key: Secret, never shared, stays on your machine.

Use case:
- [[Encryption]]: Someone uses your *public key* to encrypt a message. Only your *private* key can decrypt it. So anyone can send you a secret, but only you can read it.
- [[Cryptographic Signature]]: You use your *private* key to sign something (a message, a piece of code). Anyone with your *public* key can verify the signature is genuine, proving it came from you and it wasn't tampered with.

In the context of [[WireGuard]]/[[Secure Shell|SSH]]
- Each device generates a key pair. Devices exchange public keys. When two devices want to talk, each encrypts traffic with the other's public key — so only the intended recipient (with their private key) can read it. This is how WireGuard authenticates peers without passwords.


Asymmetric crypto used for authenticity rather than secrecy.
1. Hash your message
2. Encrypt that hash with your private key; this is the signature.
3. Recipient decrypts the signature with your public key, gets the hash
4. They hash the message themselves, and compare.

If it matches, the message came from you (only you have your private key) and wasn't modified (the hash matches). Doesn't make the message secret, just proves authorship and integrity. Used in code signing, [[Git]] commits, [[Transport Layer Security|TLS]] certificates, email (PGP).