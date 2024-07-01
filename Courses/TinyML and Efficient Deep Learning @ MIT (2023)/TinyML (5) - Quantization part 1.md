https://www.youtube.com/watch?v=TSc_BibWRhM&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=9

Recall: Pruning reduces the *number* of weights; Now, with *Quantization*, we want to reduce the number of bits required for each weight. These are orthogonal techniques that are both quite important for modern deep learning.
- If you're running on a mobile device or laptop, it's quite likely you're using a quantization technique under the hood.

![[Pasted image 20240630175558.png]]

----


Motivation: We want to make storage, memory reference, and arithmetic all cheaper!
![[Pasted image 20240630180540.png]]
The Energy decreases linearly with add, as you quantize.
For multiplication, it decreases with the square (since the work we do is quadratic)


# Numeric Data Types

![[Pasted image 20240630181538.png]]
For signed integers, we "waste" a slot
- But we have this Two's complement representation, where the first bit, instead of indicating minus, indicates *minus* 2^-7

![[Pasted image 20240630181739.png]]
We also want to represent fixed-point numbers that might not be integers (eg 3.0625)
We divide our bit array with a decimal point, where the left represents the integer, and the right represents the fraction.
(Second and third row are different methods of doing the same thing)\

Now, to the interesting part!
![[Pasted image 20240630182001.png]]
You might have heard of [[float16|FP16]], or [[bfloat16]], or FP8, etc.
32-bit is the most canonical floating point representation.
- Sign(1), Exponent(8), and Mantissa/Significant(23)

The "1+" is sort of a free lunch, because we always have this representation
These 8 bits actually appear as 2^Exponent
We get a much larger dynamic range (difference between smallest and largest value)
We subtract this Bias of 127 from the exponent...
- Using 8 bit, we can represent a 0-255 range; we use the middle point of this as the bias. This is the largest number we can represent with 7 bits.
- However many bits n you have for exponent, 2^(n-1) is the bias. This bias is fixed for a given number of bits for the exponent.

In the orange part, starting from the left, it's .5, .25, .125, and .0625 (this is the one that's marked). Generally, it's 1/2^n with base-1 indexing

So how do we represent zero? 
![[Pasted image 20240630182626.png]]
Instead of being 1 + Fraction to the power of something, we force it to have this other type of equation, where it's just *fraction* times the power of 1 - bias. We force it to be 1 minus the bias so the value is contiguous (?). With this representation, we can represent zero.
What's special about this case? The exponent is fixed

In his definition, "subnormal values" are those with the exponent fixed to zero (this is why he has the exponent section with an ice cubes in the diagrams again).

![[Pasted image 20240630183139.png]]
Representing positive and negative infinity, and NaN.

![[Pasted image 20240630183301.png]]
Normal case in Red, Subnormal case in the other rows


During training of DNNs, especailly in the first few iterations, the weights will be highly turbulent, and often gradients can get quite large. This is why large dynamic ranges are very helpful for training.
![[Pasted image 20240630183737.png]]

In practice, using [[bfloat16|BF16]] is easier to converge than using [[float16|FP16]] by avoiding extreme spikes during training (?).

![[Pasted image 20240630184259.png]]
![[Pasted image 20240630184453.png]]

Most recently, there's a new number that came out to further reduce the precision of NNs, making training faster.
During training we want higher dynamic range, but during inference, we want high precision!
- There's no free lunch: You either get higher dynamic range and less precision, or higher precision and less dynamic range.
![[Pasted image 20240630184554.png]]
==(Naming convention: "E4M3" = Exponent 4 Bits, Mantissa 3 Bits)==
- We use E4M3 for inference/forward pass, because it has smaller dynamic range but more precision.
- We use E5M2 for backward passes because it has a larger dynamic range (so we can multiply those very large gradients esp. early in training), but less precision.

---

# Quantization

![[Pasted image 20240630190715.png]]


We'll cover a number several quantization strategies.
![[Pasted image 20240630191131.png]]
We can categorize them by understanding how we do ==storage== and ==computation==.
- If the naive approach is to use F32 (full) or FP16 (half).
- We can use ==K-Means quantization== where we use integers to represent the weights, and a floating-point codebook that each weight is assigned/related to. During computing, we use floating-point arithmetic to do computing. This is used in/developed for real-time LLM inference, where the weight is the bottleneck, and compute is not -- the weights, we want to use low-precision, but the arithmetic can be in FP16 still.


[[K-Means Quantization]]
![[Pasted image 20240630191241.png]]
Say we have this matrix
![[Pasted image 20240630191251.png]]
We cluster the matrix using [[K-Means Clustering]], so that similar weights are near eachother.
![[Pasted image 20240630191313.png]]
Then we just use the centroid to represent each of them, and we just store the index in the former weight matrix, now just a cluster index. The centroids can be stored in whatever precision you'd like.

In terms of storage:
![[Pasted image 20240630191713.png]]
This is a toy example where the matrix is only 4x4; if the matrix got much larger, then the savings in the cluster index become much larger, and the cost of the codebook doesn't grow very much.
- Obviously there's a big question about how to choose your k value/size of the codebook.

How do we recover the accuracy loss when we train the model?
- We get the gradients
- We cluster teh gradients in teh same pattern as the weights, and we accumulate the same color gradients all together... sum them up or average theem, and then we get the reduced gradient. We subtract that, times the learning rate, and get the 








