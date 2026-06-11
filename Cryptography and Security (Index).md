


Terms
- [[Encryption]]: Transforming readable data into unreadable data so only parties with the proper key can recover it. Use it for confidentiality at rest or in transit, but remember it does not by itself prove identity or integrity.
- [[Symmetric Key Encryption]]: Encryption where the same secret key is used to encrypt and decrypt data. Use it for bulk data encryption because it is fast, but the shared key must be distributed and protected carefully.
- [[Asymmetric Key Encryption]]: Cryptography using a public-private key pair for encryption, key exchange, or signatures. Use it for identity, signatures, and establishing shared secrets, but avoid it for bulk data because it is much slower than symmetric encryption.
- [[Advanced Encryption Standard]] (AES): A widely used symmetric block cipher standard for encrypting data. It is usually the default choice when supported by libraries and hardware acceleration, but it must be used with a safe mode such as GCM.
- [[Rivest-Shamir-Adleman]] (RSA): A public-key cryptosystem based on the difficulty of factoring large integers. Use it mainly for compatibility with older systems, since modern elliptic-curve schemes often provide smaller keys and better performance.
- [[Key Exchange]]: A protocol that lets parties establish a shared secret over an untrusted channel. Use it when two parties need a session key, but authenticate the exchange or it can be vulnerable to man-in-the-middle attacks.
- [[Key Exchange|Elliptic Curve Diffie-Hellman]] (ECDH): A key exchange method using elliptic curve math to derive a shared secret. It is common in modern TLS and VPNs because it is efficient and supports forward secrecy when ephemeral keys are used.
- [[Cryptographic Signature]]: A value created with a private key that lets others verify authenticity and integrity with the corresponding public key. Use signatures for software updates, certificates, commits, or tokens, but do not treat them as encryption because they do not hide the signed data.
- [[Hash]]: A one-way function that maps input data to a fixed-size digest. Use hashes for integrity checks and fingerprints, but not as encryption or as password storage without a password-hashing scheme.
- [[Hash-based Message Authentication Code]] (HMAC): A keyed hash used to verify message integrity and authenticity. Use it when two parties share a secret key and need tamper detection, but it does not provide non-repudiation because both parties can produce valid MACs.
- [[Salt]]: Random data added to an input before hashing to make precomputed attacks and duplicate hashes less useful. Use a unique salt for password hashing, but combine it with a slow password-hashing function rather than a fast general hash.
- [[Transport Layer Security]] (TLS): A protocol that provides encrypted and authenticated communication over a network. Use it for most client-server network traffic, but configure certificate validation, protocol versions, and cipher suites correctly.
- [[HTTPS]]: HTTP carried over TLS to protect web traffic in transit. Use it for all public web traffic, but still validate requests and authorization because HTTPS does not make client-supplied data trustworthy.
- [[Certificate Authority]] (CA): A trusted entity that signs certificates to bind public keys to identities. Use public CAs for internet-facing services and private CAs for internal systems, but recognize that CA compromise or misissuance can undermine trust.
- [[Certificate Chain]]: A sequence of certificates linking an end-entity certificate back to a trusted root. Use it to validate certificate trust, but missing intermediates, expired certificates, or wrong hostnames can break otherwise valid deployments.
- [[Mutual TLS]] (mTLS): TLS where both client and server authenticate each other with certificates. Use it for service-to-service or high-trust client authentication, but expect certificate issuance, rotation, and debugging overhead.
- [[Secure Shell]] (SSH): A protocol for secure remote login, command execution, and tunneling. Use it for administration and developer access, but protect private keys and disable weak password-based access where possible.
- [[Virtual Private Network]] (VPN): A private network tunnel that carries traffic securely across another network. Use it to connect users, devices, or networks, but avoid making it a broad bypass around least-privilege access controls.
- [[WireGuard]]: A modern VPN protocol built around a small set of cryptographic primitives. Use it when you want a simple, fast VPN, but plan key distribution and peer configuration explicitly.
- [[Domain Name Service Security Extensions]] (DNSSEC): Extensions that use signatures to authenticate DNS records and protect against tampering. Use it when DNS authenticity matters, but it adds operational complexity and does not encrypt DNS queries.
- [[OAuth]]: An authorization framework that lets clients obtain scoped access to protected resources. Use it for delegated API access, but do not confuse it with user authentication unless it is paired with OpenID Connect.
- [[OpenID Connect]] (OIDC): An identity layer on top of OAuth 2.0 for authenticating users. Use it for single sign-on and federated login, but validate issuer, audience, nonce, and token signatures carefully.
- [[JSON Web Token]] (JWT): A compact signed token format for carrying claims between systems. Use it when services need stateless local validation, but keep lifetimes short because revocation and stale embedded claims are hard.
- [[JSON Web Key Set]] (JWKS): A JSON document containing public keys used to verify signed tokens such as JWTs. Use it to support key discovery and rotation, but cache it carefully so rotation works without causing outages.
- [[Platform-Agnostic Security Token]] (PASETO): A security token format designed to avoid common JWT cryptographic pitfalls. Use it when you control both issuer and verifier and want safer defaults, but JWT may still be easier for ecosystem interoperability.
- [[Proof Key for Code Exchange]] (PKCE): An OAuth extension that protects authorization code flows from code interception. Use it for public clients such as browser and mobile apps, and increasingly as a baseline even for confidential clients.
- [[Web Authentication API]] (WebAuthn): A browser standard for phishing-resistant public-key authentication. Use it for passkeys or hardware-backed login, but account recovery and device enrollment need careful product design.
- [[Access Token]]: A credential representing a user's authorization to access an API or resource. Use short-lived access tokens for API calls, but treat leaked bearer tokens as immediately usable by an attacker.
- [[Refresh Token]]: A longer-lived credential used to obtain new access tokens without reauthenticating. Use it to balance user experience with short-lived access tokens, but store and rotate it more carefully because compromise has a larger blast radius.
- [[Service Token]]: A credential used by a service or workload to authenticate to another system. Use it for service-to-service calls, but scope it narrowly by audience, permissions, and lifetime.
- [[Plaintext]]: Data in its original readable form before encryption. Minimize where plaintext appears in storage, logs, memory, and network boundaries because it is the form attackers actually want.
- [[Ciphertext]]: Data transformed by encryption so it is unreadable without the proper key. It is safer to store or transmit than plaintext, but metadata, access patterns, and key compromise can still leak sensitive information.
- [[Cipher]]: An algorithm for encrypting and decrypting information. Use well-reviewed standard ciphers through trusted libraries, since custom cipher design is rarely defensible.
- [[Block Cipher]]: A cipher that encrypts fixed-size blocks of data using a key. Use block ciphers through modern authenticated modes, because raw block encryption leaks structure and is not directly safe for messages.
- [[Stream Cipher]]: A cipher that encrypts data as a continuous stream, usually by combining plaintext with a keystream. Use it for high-throughput or streaming contexts, but never reuse the same key and nonce combination.
- [[ChaCha20]]: A fast stream cipher commonly used as an alternative to AES. Use it especially on devices without AES hardware acceleration, but pair it with authentication rather than using raw encryption alone.
- [[ChaCha20-Poly1305]]: An authenticated encryption scheme combining ChaCha20 encryption with Poly1305 authentication. Use it as a modern AEAD option for TLS, VPNs, and application protocols, but nonce reuse is catastrophic.
- [[Authenticated Encryption]] (AEAD): Encryption that protects both confidentiality and integrity, often with associated unencrypted metadata. Use AEAD modes by default for application encryption, because encryption without authentication is a common source of serious vulnerabilities.
- [[Message Authentication Code]] (MAC): A keyed value used to verify that a message came from someone with the key and was not modified. Use a MAC when confidentiality is not needed but integrity is, but manage the shared key as carefully as an encryption key.
- [[Nonce]]: A value intended to be used only once in a cryptographic operation. Use nonces to make repeated operations safe, but accidental reuse can completely break many cipher modes.
- [[Initialization Vector]] (IV): An input value used with a cipher mode to ensure repeated plaintexts encrypt differently. Use IVs according to the mode's requirements, because some must be random while others only need uniqueness.
- [[Session Key]]: A temporary symmetric key used to protect one communication session or transaction. Use session keys to limit damage from compromise, but derive and erase them correctly.
- [[Key Pair]]: A matched public key and private key used in asymmetric cryptography. Share the public key freely, but protect the private key because it represents identity, decryption capability, or signing authority.
- [[Key Derivation Function]] (KDF): A function that derives cryptographic keys from shared secrets, passwords, or other key material. Use KDFs to turn raw secrets into purpose-specific keys, but choose password-specific KDFs for human passwords.
- [[HMAC-Based Extract-and-Expand Key Derivation Function]] (HKDF): A KDF that extracts strong key material and expands it into one or more derived keys. Use it after key exchange or for deriving multiple protocol keys, but do not use it as a password hash.
- [[Argon2]]: A memory-hard password hashing function designed to resist brute-force attacks. Use Argon2id for new password storage when available, tuning memory and time costs to your environment.
- [[bcrypt]]: A password hashing function that uses salting and configurable work cost. Use it for broad compatibility, but be aware of password length limitations and tune the cost over time.
- [[scrypt]]: A memory-hard password-based key derivation function designed to make large-scale cracking expensive. Use it when memory hardness is useful and Argon2 is unavailable, but tune its parameters carefully.
- [[Rainbow Table]]: A precomputed table of hashes used to reverse weak or unsalted password hashes. Defend against it with unique salts and slow password hashing, though salts alone do not stop online guessing or targeted cracking.
- [[SHA-256]]: A SHA-2 hash function that produces a 256-bit digest. Use it for general integrity and content addressing, but not directly for password storage.
- [[SHA-3]]: A modern hash function family based on the Keccak sponge construction. Use it when you need a SHA-2 alternative or sponge-based construction, but SHA-2 remains more common in existing systems.
- [[BLAKE3]]: A fast cryptographic hash function designed for high performance and parallelism. Use it for high-speed checksums, content addressing, and hashing large data, but check ecosystem support before using it in protocols.
- [[MD5]]: An obsolete hash function that is no longer safe for collision-resistant security uses. Use it only for legacy checksums or non-security compatibility, never for signatures, certificates, or tamper resistance.
- [[Collision Resistance]]: The property that it is computationally infeasible to find two different inputs with the same hash. It matters for signatures and certificates, but is less central for some keyed or non-adversarial checksum uses.
- [[Preimage Resistance]]: The property that it is computationally infeasible to recover an input from its hash output. It matters when hashes represent hidden values, but weak or guessable inputs still need salting and slow hashing.
- [[Public Key Infrastructure]] (PKI): The system of certificates, authorities, policies, and processes used to manage public-key trust. Use PKI when identity must scale beyond manually exchanged keys, but expect lifecycle and governance complexity.
- [[X.509 Certificate]]: A standard certificate format used to bind a public key to an identity. Use it for TLS, mTLS, and PKI-based identity, but certificate fields, chains, and renewal are easy to misconfigure.
- [[Elliptic Curve Cryptography]] (ECC): Public-key cryptography based on elliptic curve mathematical problems. Use ECC for efficient modern key exchange and signatures, but choose well-supported curves and libraries.
- [[Curve25519]]: An elliptic curve commonly used for efficient key agreement. Use it for modern ECDH-style protocols, but it is not itself a signature scheme.
- [[Ed25519]]: A fast elliptic curve signature scheme based on Edwards curves. Use it for modern digital signatures where supported, but verify interoperability before choosing it for legacy PKI workflows.
- [[Elliptic Curve Digital Signature Algorithm]] (ECDSA): A digital signature algorithm using elliptic curve cryptography. Use it where standards and compatibility require it, but bad randomness during signing can expose the private key.
- [[Perfect Forward Secrecy]] (PFS): A property where past sessions remain protected even if a long-term private key is later compromised. Prefer protocols with PFS for network sessions, but remember it does not protect live sessions or compromised endpoints.
- [[Transport Layer Security|TLS]] Handshake: The negotiation phase where TLS peers authenticate, agree on parameters, and derive session keys. Understand it when debugging HTTPS, mTLS, or performance, because certificate validation and round trips often dominate connection setup.
- [[End-to-End Encryption]] (E2EE): Encryption where only the communicating endpoints can read the protected content. Use it when intermediaries should not access content, but it complicates moderation, search, recovery, and server-side processing.
- [[Encryption at Rest]]: Encryption applied to stored data such as disks, databases, backups, or object storage. Use it to reduce exposure from lost media or storage compromise, but it is only as strong as key management and access control.
- [[Encryption in Transit]]: Encryption applied to data while it moves across a network. Use it for service, user, and administrative traffic, but it does not protect data once it reaches either endpoint.
- [[Key Rotation]]: The practice of replacing cryptographic keys on a planned or event-driven schedule. Use it to limit exposure and recover from suspected compromise, but design for overlapping validity periods to avoid breaking clients.
- [[Key Management Service]] (KMS): A service for creating, storing, controlling, and using cryptographic keys. Use it to centralize key policy and auditability, but account for dependency, latency, and provider trust.
- [[Hardware Security Module]] (HSM): Dedicated hardware for protecting and performing operations with sensitive cryptographic keys. Use it for high-value keys and compliance-sensitive systems, but expect cost, operational friction, and throughput limits.
- [[Envelope Encryption]]: A pattern where data is encrypted with a data key, and that data key is separately encrypted with a higher-level key. Use it to encrypt large amounts of data while centralizing master-key control, but track data keys and rotation metadata carefully.














