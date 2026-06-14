---
aliases:
  - Public Key
  - Private Key
  - Asymmetric Encryption
  - Asymmetric Key Cryptography
  - Asymmetric Cryptography
---
Uses a key pair (Public key, Private key) for encryption and decryption, unlike [[Symmetric Key Encryption]].

Uses two mathematically related keys that are generated together:
- Public Key 
	- Used to encrypt data. (Or, in the case of [[Signature]]s, used to *verify* a signature)
	- Safe to distribute, can be placed in [[Certificate]]s, sent over the network, published in [[Domain Name Service|DNS]], or embedded in software.
- Private Key
	- Used to decrypt data. (Or, in the case of [[Cryptographic Signature|Signature]]s, used to *sign* signatures)
	- Must remain secret; anyone who has access can impersonate the owner or decrypt data intended for them (depending on how the key is used).

Simplified Flow:
```
Bob generates:
public key + private key

Bob shares public key with Alice
Bob keeps private key secret

Alice:
plaintext + Bob's public key -> ciphertext

Bob:
ciphertext + bob's private key -> plaintext
```

# Uses of Asymmetric Cryptography
1. Public-Key Encryption: Used to send confidential data to someone.
```
Encrypt with the recipient's public key
Decrypt with the recipient's private key
```


2. Digital Signatures: Used to prove who created or approved data.
```
Sign with the sender's private key
Verify with the sender's public key
```
- Signing usually works like: message -> hash function -> message digest  -> signature algorithm using private key -> signature.
- Verification usually works like: message + signature + public key -> hash the message again -> check signature against digest and public key -> valid or invalid.
The hash function matters because signature algorithms usually sign a fixed-size digets, not an arbitrary-length message directly.


3.  Key Agreement: Used to establish a shared [[Symmetric Key Encryption|Symmetric Key]] over an insecure network (e.g. [[Key Exchange|Diffie-Helman]]/[[Key Exchange|Elliptic Curve Diffie-Hellman]]])
```
Alice: Alice's private key + Bob's public key
Bob: Bob's private key + Alice's public key

Both are able to derive the same shared secret, a symmetric session key, without transmitting it!
```

# Use with [[Symmetric Key Encryption]]
- In practice, systems often use hybrid encryption: Asymmetric cryptography is used to first solve identity and key exchange, then symmetric encryption is used for the actual data.

In a Typical [[Transport Layer Security|TLS]]-style flow:
1. Client connects to server
2. Server proves identity using a [[Certificate]] and [[Asymmetric Key Encryption|Asymmetric Key Cryptography]] (Public/Private)
3. Client and server perform a key agreement
4. Both derive the same [[Symmetric Key Encryption|Symmetric Key]] for their session
5. All further traffic is encrypted using fast, symmetric encryption.
This is how [[HTTPS]] avoids using slower asymmetric encryption for *every message* during a long TLS session in which many HTTP messages are exchanged.

Typically:
- Use [[Symmetric Key Encryption|Symmetric Encryption]] for encrypting actual data (fast).
- Use asymmetric cryptography to:
	- Exchange symmetric keys  (or the data used to generate them)
	- Verify identity
	- Create digital signatures
	- Establish secure sessions
- Modern secure systems usually combine both: Symmetric encryption provides speed, while asymmetric cryptography provides scalable trust and key establishment.





__________ 

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