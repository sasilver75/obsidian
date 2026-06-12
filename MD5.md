

- Digest size: 128 bits 
- Status: Cryptographically broken!


# Why it's no longer Cryptographic
- Originally designed as a [[Hash|Cryptographic Hash Function]], but no longer considered secure for cryptographic purposes due to several security vulnerabilities. with MD5, researchers can intentionally create different files or messages that generate teh exact same hash in a matter of seconds or hours.

# Where it can still be used
- Completely unsafe for security-related use-cases like password storage or digital [[Certificate]]s.
- Can still be used for non-cryptographic purposes:
	- File Checksums: You can use it to verify data integrity against accidental file corruption during downloads or transfers.
	- Database Keys: It can be used as a quick way to partition data or distribute keys in certain non-security-critical databases.

# Recommended Secure Alternatives
- For security, experts recommend something in the SHA-2 or SHA-3 families such as [[SHA-256]].
- For password hashing specifically, algorithms like [[bcrypt]] or [[Argon2]] are the industry standard.





