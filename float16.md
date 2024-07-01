---
aliases:
  - FP16
---
Along with [[bfloat16]], often referred to as "half precision"

- ==Floating point numbers== are predominantly used, because they can represent a wide range of values with high precision. The *n* bits of a FP number are divided into three distinct components:
	- ==Sign==: A single bit; 0 indicates a positive, and 1 signals a negative number.
	- ==Exponent== (impacts range): A segment of bits that represents the power to which the base (usually 2, in binary representation) is raised. ***Can also be positive or negative, allowing the number to represent very large or very small values.***
	- ==Significand/Mantissa== (impacts precision): The remaining bits; represents the significant digits of the number. ***The precision of a number heavily depends on the length of the significand.***

==[[float16|FP16]]== uses 16 bits to represent a number
- 1 bit for the sign
- 5 bits for the exponent
- 10 bits for the significand
More memory efficient and accelerates computation, but the reduced range and precision can impact numerical stability, potentially impacting the resulting model accuracy.

![[Pasted image 20240627113928.png]]
![[Pasted image 20240618105044.png]]
![[Pasted image 20240630183725.png]]