---
aliases:
  - MAC
---
A short cryptographic value computed from a message and a shared secret key, used to verify that the message has not been modified and that it was produced by someone who knows the secret key.
> A tamper-evident seal made with a shared secret. Anyone can see the sealed message (if you don't separately encrypt it), but only parties who know the secret key can create or verify a seal.

Solves two problems:
1. Message integrity: "Did the message change in transit/storage?"
2. Symmetric-key message authentication: "Was this message produced by someone who knows the shared secret key?"
It does NOT provide confidentiality itself; the message can still be readable unless the message is separately encrypted.


Suppose server sends:
```
message = "transfer $100 to account 123"
Authentication Tag = MAC(secret_key, message)
```

The receiver them recomputes the MAC over the received message using the same secret key (that was previously exchanged, likely using [[Asymmetric Key Encryption|Asymmetric Cryptography]] for the key exchange).

If an attacker had changed the message to:
```
transfer $900 to account 123
```
Then the old Authentication Tag no longer verifies, because the attack can't compute a valid new Authentication Tag without the secret key.


# Common MAC constructions:
- [[Hash-based Message Authentication Code]] (HMAC): A MAC built from a cryptographic hash functions, such as HMAC-SHA-256.
- [[Poly1305]]: A fast MAC often paired with [[ChaCha20]] in [[ChaCha20-Poly1305]] for [[Authenticated Encryption with Associated Data]].