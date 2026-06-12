---
aliases:
  - Symmetric Encryption
  - Symmetric Key
  - Symmetric Key Cryptography
  - Symmetric Cryptography
---
Uses ==one shared secret key== for ***both*** [[Encryption]] and decryption, unlike [[Asymmetric Key Encryption]]

Simplified Flow:
- Sender and receiver both have the same secret key 
	- Often, this exchange is accomplished via [[Asymmetric Key Encryption|Asymmetric Encryption]] means.
- Sender: `plaintext + secret key -> ciphertext`
- Receiver: `ciphertext + secret key -> plaintext`


Detailed Explanation:
- In symmetric encryption, both parties must already know the same secret key before secure communication can happen. 
- For modern systems, symmetric encryption should usually be authenticated encryption, often called [[Authenticated Encryption with Associated Data]] (AEAD). 

Example:
The Sender uses:
[[Plaintext]] + [[Symmetric Key Encryption|Symmetric Key]] + [[Nonce]] = [[Ciphertext]] + [[Authentication Tag]]
The Receiver uses:
[[Ciphertext]] + [[Authentication Tag]] + same [[Symmetric Key Encryption|Symmetric Key]] + [[Nonce]] = [[Plaintext]], if authentication succeeds

==If the ciphertext or tag has been modified, decryption fails.==
This is important because encryption alone only hides data. It doesn't necessarily prove the data was *not* changed. [[Authenticated Encryption with Associated Data|AEAD]] modes like `AES-GCM` or `ChaCha20-Poly1305` provide both confidentiality and integrity.


# Strengths and Weaknesses
Strengths
- Very fast
- Efficient for large amounts of data
- Simple cryptographic model, once the key is shared
Weaknesses:
- The hard problem is key distribution, the [[Key Exchange|Key Exchange Problem]].
	- If Alice and Bob need the same secret key, how do they share it securely before they already have a secure channel?
	- This is where [[Asymmetric Key Encryption|Asymmetric Key Cryptography]] is often used.

