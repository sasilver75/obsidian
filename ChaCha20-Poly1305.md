An [[Authenticated Encryption with Associated Data|AEAD]] construction that is a modern alternative to [[AES-GCM]], especially attractive when speed, software performance, and mobile compatibility matter.

[[ChaCha20]]: A symmetric stream cipher. Encrypts data by generating a pseudorandom keystream and XORing it with the plaintext. Provides confidentiality
```
key + nonce + counter -> keystream
plaintext XOR keystream -> ciphertext
```

[[Poly1305]]: A [[Message Authentication Code]] (MAC), produces an [[Authentication Tag]] that verifies message integrity and authenticity. Helps prevent tampering.
```
message + MAC key -> authentication tag
```

They can be used together to make [[ChaCha20-Poly1305]], an [[Authenticated Encryption with Associated Data]]  (AEAD) construction, providing:
- ==Confidentiality==: Attackers can't read the plaintext
- ==Integrity==: Attackers can't modify the ciphertext undetected
- ==Authentication==: Receivers know that the message came from someone with the key

Use cases:
- [[Transport Layer Security|TLS]]/[[HTTPS]]: Encrypts web traffic after session keys are established.
- [[QUIC]]/[[HTTP 3]]: Common AEAD option for modern transport security.
- [[Virtual Private Network|VPN]]s: Used in systems like [[WireGuard]] for fast, secure packet transmission