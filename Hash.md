---
aliases:
  - Hash Function
---
A one-way only function; input -> fixed-size output (digest).
Same input always produces same output. Any change to the input completely changes the output.

Used for:
- Integrity checking: Hash a file, send the hash separately. Recipient hashes the file they got; if it matches, nothing was tampered with.
- Password storage: Never store passwords in plaintext, always store the hash. Use [[Salt]]ing to prevent [[Rainbow Table]] attacks.
- Digital signatures: Sign the hash of a document, not the whole document.

Common algorithms include: [[SHA-256]], [[MD5]], etc.

