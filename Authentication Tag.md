

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