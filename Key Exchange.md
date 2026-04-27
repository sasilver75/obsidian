---
aliases:
  - Diffie-Helman
  - Elliptic Curve Diffie-Hellman
  - ECDH
---
How two parties agree on a shared secret over a public channel without ever sending the secret.


The classic analogy is mixing paint colors:
1. You and I both start with yellow (public)
2. I add my secret color, and you add yours; we each have different mixtures.
3. We swap mixtures publically.
4. I add my secret to your mixture, and you add your secret to my mixture; we both arrive at the same final color.
5. An observer only sees the initial yellow and the two intermediate mixtures being exchanged, but can't reconstruct the final color without knowing one of the secrets.


The modern variant is [[Key Exchange|Elliptic Curve Diffie-Hellman]], the same idea on elliptic curves.
- Used in [[Transport Layer Security|TLS]], [[WireGuard]], Signal.






