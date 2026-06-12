---
aliases:
  - AEAD
---
An encryption construction that not only encrypts the message, but also proves that the encrypted message hasn't been tampered with. AEAD is the modern, safe default for **symmetric encryption of actual messages**, especially after some other mechanism, often asymmetric key exchange, has established the shared secret key.

Taxonomically, AEAD lives inside symmetric cryptography. More specifically, it's the modern default kind of symmetric encryption scheme used when you want:
1. ==Confidentiality==: The [[Plaintext]] is hidden.
2. ==Integrity==: Attackers can't modify the [[Ciphertext]] undetected; receiver can tell if ciphertext was modified.
3. ==Authentication==: Receivers know that the message came from someone with the key
4. Associated Data protection: Some metadata can remain unencrypted but still be protected against tampering.

> A symmetric-key encryption construction that encrypts data and authenticates both the encrypted data and optional unencrypted metadata.

> It's closer to:
> AEAD = symmetric encryption + Message Authentication Code, packaged as one safe construction

Encryption alone only answers: "Can the attacker read the message?"
It does not necessarily answer: "Did an attacker modify the message?"
AEAD is the modern answer.

On Encryption
```
plaintext + key + nonce + associated data -> ciphertext + Authenticaiton Tag
```
On Decryption
```
ciphertext + Authentication Tag + key + nonce + same associated data -> plaintext
```
If the Authentication Tag verification fails, the receiver must reject the message and not use the plaintext.


Common AEAD algorithms include [[AES-GCM]] and [[ChaCha20-Poly1305]].

Think of an AEAD message as having two regions:
- Encrypted and Authenticated: `Plaintext -> Ciphertext`
- Not Encrypted, but Authenticated: `Associated Data`
	- This associated data is not secret, but it is security-critical.
	- For example, a network protocol might encrypt the message body, but leave routing information visible.

An AEAD encryption function usually has this shape:
```
Encrypt(key, nonce, plaintext, associated_data) -> ciphertext, authentication tag
```
while the corresponding decryption function looks like:
```
Decrypt(key, nonce, ciphertext, associated_data, authentication_tag) -> plaintext OR fail
```

The receiver recompute or verifies the [[Authentication Tag]] using the same key, nonce, ciphertext, and Associated Data. If any of those inputs differ from what the sender used, verification fails.

AEAD usually requires a [[Nonce]], sometimes called an [[Initialization Vector]] (IV) depending on the construction and API. The nonce does not need to be secret, but it must usually be unique for a given key (and used once).

Critical rule:
> If Authentication Tag verification fails, the decrypted plaintext must be rejected and not used.




In a secure connection like [[HTTPS]]:
- [[Asymmetric Key Encryption|Asymmetric Cryptography]] is used to solve the bootstrapping and do [[Key Exchange]], during something like the [[Transport Layer Security|TLS Handshake]].
	- The server proves identity using [[Certificate]]s and [[Asymmetric Key Encryption|Asymmetric Cryptography]].
	- The client and server perform key exchange and derive shared symmetric session keys.
- After that, the actual application data is usually protected with [[Symmetric Key Encryption|Symmetric Cryptography]], specifically [[Authenticated Encryption with Associated Data|AEAD]]:
```
HTTP request/response data
    -> encrypted and authenticated with AES-GCM or ChaCha20-Poly1305
```


# Why does Associated Data Exist?
Many real systems have metadata that must remain visible, but still must not be tampered with:

| Scenario                                               | Plaintext            | Associated Data                                |
| ------------------------------------------------------ | -------------------- | ---------------------------------------------- |
| [[Transport Layer Security\|TLS]]-like record protocol | Application data     | Record type, protocol version, sequence number |
| Encrypted database field                               | Sensitive value      | Row ID, column name, schema version            |
| API token                                              | Claims or token body | Token version, issuer, key identifier          |
| File encryption                                        | File contents        | Filename, file type, creation metadata         |
| Messaging app                                          | Message body         | Conversation ID, sender ID, message number     |

Associated data is data that is:
1. Not encrypted
2. Still nees to be authenticated
3. Required to match during decryption




