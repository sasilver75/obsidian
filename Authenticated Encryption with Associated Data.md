---
aliases:
  - AEAD
---
An encryption constructions that not only encrypts the message, but also proves that the encrypted message hasn't been tampered with.
A type of encryption providing both:
1. ==Confidentiality==: The [[Plaintext]] is hidden.
2. ==Integrity==: Attackers can't modify the [[Ciphertext]] undetected; receiver can tell if ciphertext was modified.
3. ==Authentication==: Receivers know that the message came from someone with the key

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

Critical rule:
> If Authentication Tag verification fails, the decrypted plaintext must be rejected and not used.




# Why does Associated Data Exist?
Many real systems have metadata that must remain visible, but still must not be tampered with:

| Scenario                                               | Plaintext            | Associated Data                                |
| ------------------------------------------------------ | -------------------- | ---------------------------------------------- |
| [[Transport Layer Security\|TLS]]-like record protocol | Application data     | Record type, protocol version, sequence number |
| Encrypted database field                               | Sensitive value      | Row ID, column name, schema version            |
| API token                                              | Claims or token body | Token version, issuer, key identifier          |
| File encryption                                        | File contents        | Filename, file type, creation metadata         |
| Messaging app                                          | Message body         | Conversation ID, sender ID, message number     |




