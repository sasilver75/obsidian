---
aliases:
  - Hash Function
  - Cryptographic Hash Function
---
A one-way only function that takes input of any size and produces a fixed-size output (called a hash, digest, or fingerprint).
- Same input always produces same output. Any change to the input completely changes the output.

Used for:
- Integrity checking: Hash a file, send the hash separately. Recipient hashes the file they got; if it matches, nothing was tampered with.
- Password storage: Never store passwords in plaintext, always store the hash. Use [[Salt]]ing to prevent [[Rainbow Table]] attacks.
- Digital signatures: Sign the hash of a document, not the whole document.

Common algorithms include: [[SHA-256]], [[MD5]], etc.
 


A **[[Hash|Hash Function]]** is any function that maps input data to a fixed-size value.
- CRC32, xxHash, [[MurmurHash]], CityHash, FarmHash
- Used for hash tables, [[Checksum]]s, [[Sharding]], [[Cache|Caching]]
A **[[Hash|Cryptographic Hash Function]]** is a special kind of hash function designed to resist attackers.
- [[SHA-256]], [[SHA-512]], [[SHA-3]], [[BLAKE2]], [[BLAKE3]]
- Used for integrity, [[Signature]]s, [[Hash-based Message Authentication Code|HMAC]], content IDs.
# [[Hash|Cryptographic Hash Function]]s
A cryptographic hash function should provide:
- Deterministic
	- Same input gives same hash
- Fixed-size output
	- Output length is constant, regardless of input size
- Preimage resistance
	- Given a hash, it should be infeasible to recover the original input
- Second-preimage resistance
	- Given an input, it should be infeasible to find another input with the same hash
- Collision resistance
	- It should be infeasible to find any two different inputs with the same hash
- Avalanche effect
	- A tiny input change should produce a very different hash

Important: Hashing ==is not encryption;== Hashing is one-way, and has no decryption key.


# Common Cryptographic Hash Functions (and their security status)
Note that there are still use cases for even non-cryptographic hash functions.
- [[MD5]]: 128 bit digest, Broken security status. Fast, but collision attacks work.
- [[SHA-1]]: 160 bits, Broken security status. Legacy only, collision attacks are practical.
- [[SHA-256]]:  256 bits, Secure. Very common, widely trusted.
- [[SHA-512]]: 512 bits, Secure. Often faster than SHA-256 on 64-bit CPUs
- [[SHA-3]]: 224-512 bits, Secure. Different design from SHA-2
- [[BLAKE2]]: 224-512 bits, Secure. Fast, modern, simpler than SHA-2 in many contexts
- [[BLAKE3]]: 256 bits by default, Secure, modern. Very fast, parallelizable

Special cryptographic hash-functions that are designed specifically for storing and verifying passwords:
- [[bcrypt]]
- [[Argon2]]
These are intentionally slow (so that they're expensive to brute-force), each using a unique [[Salt]].
`password + salt + cost parameters -> bycrypt/Argon2 -> password hash`



