---
aliases:
  - Hex
---

Binary numbers get very long very quickly, making it hard for people to deal with.

==One digit in Hex maps to exactly 4 bits ( a nibble)==, so ==a byte is always two hex digits==.

That makes binary data compact and readable: 0xFF is cleaner than 0b111111 or 255.
- It's why colors, memory addresses, and file magic numbers are all conveniently written in hex.

```
0 1 2 3 4 5 6 7 8 9 a b c d e f

e.g.
2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824

0 = 0000
1 = 0001
2 = 0010
...
9 = 1001
a = 1010
b = 1011
c = 1100
d = 1101
e = 1110
f = 1111

Two hexadecimal characters represent one byte:
ff = 11111111 = 255
0a = 00001010 = 10
2c = 00101100 = 44

A SHA-256 cryptographic hash function gives a 256-bit hash digest output, often written as 256/4 = 64 hexadecimal characters: 
9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
```








