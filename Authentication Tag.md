A short, cryptographic value attached to encrypted or authenticated data so that the receiver can verify that the data was produced by someone who knew the correct secret key and was not modified in transit.

Often appears in [[Symmetric Key Encryption|Symmetric Cryptography]], in the context of [[Authenticated Encryption with Associated Data]] (AEAD).

> The Authentication Tag is like a tamper-evident seal computed from: the secret key, the [[Nonce]] (or [[Initialization Vector|IV]]), the [[Ciphertext]], and sometimes unencrypted but protected metadata, called *Additional Authenticated Data*.

The Authentication Tag itself is not secret; Attackers can/may see it. The security property is that attackers should not be able to create a valid Authentication Tag for modified data without knowing the secret key.


Example, in [[AES-GCM]]
```
plaintext:  "transfer $100"
AAD:        "from=alice,to=bob"
nonce:      random or unique value
key:        shared secret key

output:
ciphertext: encrypted bytes
Authentication Tag: 128-bit value authenticating ciphertext + AAD
```
- If the attacker:
	- Changes the ciphertext so that the decrypted message would become `transfer $900` ...
	- Changes the unencrypted metadata from `to=bob` to `to=eve`...
... then the Authentication Tag verification will fail.

The Authentication tag provides ==integrity== and ==authenticity==, not secrecy. The Ciphertext provides secrecy.
The Authentication tag says, to the receiver:
> "This exact Ciphertext and protected metadata match what someone with the secret key produced."



# Comparison with [[Message Authentication Code]]
- A [[Message Authentication Code|MAC]] is the broader cryptographic idea of using a secret key to compute a short verification value over some data, so that a receiver with the same secret key can detect modification and confirm that some key-holder produced it.
- An [[Authentication Tag]] is usually the specific verification value that's output alongside ciphertext by an authenticated encryption scheme, such as [[AES-GCM]].
	- Authentication Tags are MAC-like values, but (in a scheme like [[Authenticated Encryption with Associated Data|AEAD]]), they're integrated into the encryption mode and bound together with the key/nonce/IV/ciphertext/AAD... while a standalone MAC such as [[Hash-based Message Authentication Code|HMAC]] authenticates arbitrary bytes without encrypting them.

# Comparison with [[Signature]]
- They look similar at a distance because both are extra bytes to a message to prove "this message checks out," but they rely on different key models and give different guarantees.
- With an Authentication Tag, verification requires *the same secret key* used to create it. ([[Symmetric Key Encryption|Symmetric Encryption]])
- With a Signature, signing and verification use *different keys*. ([[Asymmetric Key Encryption|Asymmetric Encryption]])
- Example:
	- You might use an (e.g. [[Hash-based Message Authentication Code|HMAC]]-generated) Authentication Tag for a webhook verification, proving that the incoming payload came from someone with the shared webhook secret key.
	- You might use a Signature when a project signs a release artifact with a private key; anyone can then verify the release using the project's public key. 
- Note: People often say "signature" for both. e.g. in our HMAC webhook example. they might call that authentication tag a "request signature," informally.