---
aliases:
  - BF16
---
Google Brain Floating Point Format
Along with [[float16|FP16]], often referred to as "half precision"

- ==Floating point numbers== are predominantly used, because they can represent a wide range of values with high precision. The *n* bits of a FP number are divided into three distinct components:
	- ==Sign==: A single bit; 0 indicates a positive, and 1 signals a negative number.
	- ==Exponent== (impacts range): A segment of bits that represents the power to which the base (usually 2, in binary representation) is raised. ***Can also be positive or negative, allowing the number to represent very large or very small values.***
	- ==Significand/Mantissa== (impacts precision): The remaining bits; represents the significant digits of the number. ***The precision of a number heavily depends on the length of the significand.***

==[[bfloat16|BF16]]== also uses 16 bits to represent a number
- 1 bit for the sign
- 8 bits for the exponent
- 7 bits for the significand
Compared to [[float16|FP16]], BF16 has a larger representable range, which decreases underflow and overflow risks, but has reduced precision due to fewer significand bits. BF16 typically doesn't significantly impact model performance, and is a useful compromise for deep learning tasks.



![[Pasted image 20240627113928.png]]
![[Pasted image 20240618105044.png]]
![[Pasted image 20240630183729.png]]
