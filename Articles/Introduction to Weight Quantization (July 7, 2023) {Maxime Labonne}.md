Author: [[Maxime Labonne]]
https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c

-----

LLMs are known for being computationally expensive -- the size of the model is typically calculated by multiplying the *number of parameters* (size) by the *precision of these values* (data type).
- To save memory, weights can be stored using ==lower-precision datatypes== through a process known as [[Quantization]].

We distinguish two main families of weight quantization techniques:
1. [[Post-Training Quantization]] (PTQ): The weights of an *already trained model* are converted to a lower-precision representation *without any retraining*. This is easy to implement, but it comes with a potential performance loss.
2. [[Quantization-Aware Training]] (QAT): Incorporates the weight conversion process *during the pre-training or fine-tuning stage*, resulting in enhanced model performance relative to PTQ. However, QAT is computationally expensive and demands representative training data (?).

## Background on Floating Point Representation
- The choice of precision/data type dictates the quantity of computational resources required, and affects both speed and efficiency of the model. Balancing speed/cost and performance is important in production.
- ==Floating point numbers== are predominantly used, because they can represent a wide range of values with high precision. The *n* bits of a FP number are divided into three distinct components:
	- ==Sign==: A single bit; 0 indicates a positive, and 1 signals a negative number.
	- ==Exponent== (impacts range): A segment of bits that represents the power to which the base (usually 2, in binary representation) is raised. ***Can also be positive or negative, allowing the number to represent very large or very small values.***
	- ==Significand/Mantissa== (impacts precision): The remaining bits; represents the significant digits of the number. ***The precision of a number heavily depends on the length of the significand.***

![[Pasted image 20240627113306.png]]

The most commonly-used datatypes in Deep Learning that we'll dive into are:
- [[float32]] (FP32)
- [[float16]] (FP16)
- [[bfloat16]] (BF16)

==[[float32|FP32]]== uses 32 bits to represent to represent a number
- 1 bit for the sign
- 8 bits for the exponent
- 23 bits for the significand
This provides a high degree of precision, but it has high memory and computational footprint.

==[[float16|FP16]]== uses 16 bits to represent a number
- 1 bit for the sign
- 5 bits for the exponent
- 10 bits for the significand
More memory efficient and accelerates computation, but the reduced range and precision can impact numerical stability, potentially impacting the resulting model accuracy.

==[[bfloat16|BF16]]== also uses 16 bits to represent a number
- 1 bit for the sign
- 8 bits for the exponent
- 7 bits for the significand
Compared to FP16, BF16 has a larger representable range, which decreases underflow and overflow risks, but has reduced precision due to fewer significand bits. BF16 typically doesn't significantly impact model performance, and is a useful compromise for deep learning tasks.

![[Pasted image 20240627113916.png|500]]

In ML jargon, FP32 is often termed “full precision” (4 bytes), while BF16 and FP16 are “half-precision” (2 bytes).

Even smaller is the [[INT8]] data type, which consists of an 8-bit representation capable of storing 2⁸ = 256 different values. In the next section, we’ll see how to convert FP32 weights into an INT8 format.


## Naive 8-bit Quantization
- Let's look at two quantization techniques that we'll use to map an FP32 tensor to an INT8 tensor:
	- A ==symmetric== one with ==absolute maximum (absmax) quantization==
	- An ==asymmetric== one with ==zero-point quantization==

With [[Absolute Maximum Quantization|Absmax Quantization]]:
- The original number is divided by the absolute maximum value of the tensor, and then multiplied by a scaling factor (127) to map inputs into the range \[-127, 127\] (for our 8 bit quantization example), before being rounded to the nearest number (we rearrange these terms a bit below).
- If we want to recover the original FP16 values, we divide out INT8 number by the quantization factor, acknowledging some loss of precision due to rounding.
- Example
	- If we had an absolution maximum value of 3.2, a weight of 0.1 would be quantized to ROUND((127/3.2)\*0.1) = ROUND(3.968) = 4
![[Pasted image 20240627120852.png|300]]
```python
def absmax_quantize(X):
	# Calculate scaling factor
	scale = 127/torch.max(torch.abs(X))
	# Quantize
	X_quant = (scale * X).round()
	# Dequantize
	X_dequant = X_quant / scale

	return X_quant.to(torch.int8), X_dequant
```

With [[Zero-Point Quantization]]:
- We can consider asymmetric input distributions, which is useful when considering (eg) the output of a [[Rectified Linear Unit|ReLU]] function. 
- The input values are first scaled by the total range of values (255) divided by the *difference between the maximum and minimum values*.
	- This distribution is then shifted to map it into the range \[-128, 127\] (notice the extra value, compared to [[Absolute Maximum Quantization|Absmax Quantization]]).
![[Pasted image 20240627120838.png|400]]
Then, we can use these variables to quantize or dequantize our weights:
![[Pasted image 20240627120940.png|400]]
Example
- If we have a maximum value of 3.2 and a minimum value of -3.0, we calculate the *scale* as $255/(3.2+3.0) = 41.13$, and the *zero-point* as $-round(41.13 * -3.0) - 128 = -5$, so our previous weight of 0.1 would be quantized to $round(41.13*0.1-5) = -1$.
- This -1 is very different form the previous value 4 obtained using [[Absolute Maximum Quantization|Absmax Quantization]].
```python
def zeropoint_quantize(x):
	# Calculate the value range (denominator)
	x_range = torch.max(X) - torch.min(X)
	x_range = 1 if x_range == 0 else x_range

	# Calculate scale
	scale = 255 / x_range

	# Shift by zero-point
	zeropoint = (-scale * torch.min(X) - 128).round()

	# Quantize: scale and round the inputs
	X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

	# Dequantize
	X_dequant = (X_quant - zeropoint) / scale

	return X_quant.to(torch.int8), X_dequant
```


![[Pasted image 20240627121601.png]]



Let's use these on a real model!
We load the model and tokenizer or GPT-2 -- this is probably a model that's small enough that we wouldn't want to quantize it in reality, but it can be good enough for this tutorial!

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

# Set device to CPU for now
device = "cpu"

# Load model and tokenizer
model_id = "gpt2"
model = AutoModelForCausalML.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print the model initial size
print(f"Model size: {model.get_memory_footprint():,} bytes")
Model size: 510,342,192 bytes
```

So the model size of GPT-2 is initially 487MB in FP32 full precision. 
Let's quantize it using our two methods (absmax, zero-point) and see the results!

We apply the technique to the first attention layer of GPT-2 to see the results:

```python
# Extract the weights of the first layer
weights = model.transformer.h[0].attn.c_attn.weight.data
print("Original weights: ", weights)

# Quantize the layer using absmax quantization
weights_abs_quant, _ = absmax_quantize(weights)
print(weights_abs_quant)

# Quantize using zeropoint quantization
weights_zp_quant, _ = zeropoint_quantize(weights)
print(weights_zp_quant)
```
![[Pasted image 20240627122623.png|400]]
The difference between the original and quantized INT8 values is clear, but the difference between the absmax and zero-point quantization strategies is more subtle -- it seems like the values are shifted by a value of -1, suggesting that the weight distribution in this layer is quite symmetric.

We can compare these techniques by quantizing every layer in GPT-2 and creating two new models (model_abs and model_zp). 
- Specifically: *We're going to replace the original weights with dequantized ones, because PyTorch doesn't allow INT8 Matmul by default*. We would dequantize them to *run the model*, but *store them* in INT8.
	- Later, we'll use [[bitsandbytes]] to solve this issue.

```python
import numpy as np
from copy import deepcopy

# Store original weights
weights = [param.data.clone() for param in model.parameters()]



# Quantize all model weights using absmax
model_abs = deepcopy(model)
weight_abs = []
for param in model_abs.parameters():
	_, dequantized = absmax_quantize(param.data)
	param.data = dequantized
	weights_abs.append(dequantized)

# Quantize all model weights using zerio point
model_zp = deepcopy(model)
weights_zp = []
for param in model_zp.parameters():
	_, dequantized = zeropoint_quantize(param.data)
	param.data = dequantized
	weights_zp.append(dequantized)

# Let's check the impact of this process by plotting the distribution of the dequantized and quantized weights
```

![[Pasted image 20240627123134.png|400]]
Both plots look pretty similar, with a surprising spike around 0! This spike shows that our quantization is quite lossy, since reversing the process didn't output the original weight values.
- This is particularly true for absmax, which has both a lower valley and a higher spike around 0.

Let's compare the performance of the original and quantized models, using a generate_text() function to generate 50 tokens using [[Top-K Sampling]].
```python
def generate_text(model, input_text, max_length=50):
	# tokenize our inputs
	input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
	# generate the output
	output = model.generate(
		inputs=input_ids,
		max_length=max_length,
		do_sample=True,  # Whether to use sampling; greedy decoding otherwise 
		top_k=30,  # number of highest-probability vocab tokens to sample from
		pad_token_id=tokenizer.eos_token_id,
		attention_mask=input_ids.new_ones(input_ids.shape) # New ones returns a tensor of the same shape, filled with ones
	)
	# reutnr the decoded tokens
	return tokenizer.decode(output[0], skip_special_tokens=True)

original_text = generate_text(model, "I have a dream")
absmax_text = generate_text(model_abs, "I have a dream")
zp_text = generate_text(model_zp, "I have a dream")

Original model:  
I have a dream, and it is a dream I believe I would get to live in my future. I love my mother, and there was that one time I had been told that my family wasn't even that strong. And then I got the  
--------------------------------------------------  
Absmax model:  
I have a dream to find out the origin of her hair. She loves it. But there's no way you could be honest about how her hair is made. She must be crazy.  
  
We found a photo of the hairstyle posted on  
--------------------------------------------------  
Zeropoint model:  
I have a dream of creating two full-time jobs in America—one for people with mental health issues, and one for people who do not suffer from mental illness—or at least have an employment and family history of substance abuse, to work part

```

Above: Instead of manually trying to see if one output makes more sense than the others, we can quantify it by calculating the perplexity of each output.
- Measures the uncertainty of a model in predicting the next token in a sequence.
- (Note that in reality a sentence with high perplexity oculd also be correct)

```python
# a minimal implementation
def calculate_perplexity(model, text):
	# Encode the text
	encodings = tokenizer(text, return_tensors="pt").to(device)

	# Define input_ids and target_ids
	input_ids = encodings.input_ids
	target_ids = input_ids.clone()

	with torch.no_grad():  # Should probably be using torch.inference_mode these days
		outputs = model(input_ids, labels=target_ids)  # The labels are what is used to calculate loss against. By giving the same sequence as both input and labels, the model is essentially being asked to predict each token, given all of the previous tokens in the sequence.

	# Loss calcualtion
	neg_log_likelihood = outputs.loss

	# Perplexity
	ppl = torch.exp(neg_log_likelihood)

	return ppl

ppl = calculate_perplexity(model, original_text)  
ppl_abs = calculate_perplexity(model_abs, absmax_text)  
ppl_zp = calculate_perplexity(model_zp, absmax_text)

Original perplexity:  15.53  
Absmax perplexity:    17.92  
Zeropoint perplexity: 17.97
```

We see that the perplexity of the original model is slightly lower than the two others! We would repeat this experiment multiple times to see the average difference between models, in reality.
- In theory, ==zero-point quantization should be slightly better than absmax, but is also more costly to compute.==

In this example, we applied quantization to entire layers, but we could apply it at different granularity levels!
- In practice, we ==often prefer vector-wise quantization==, which considers variability in values in rows and columns inside the same tensor.

But even vector-wise quantization doesn't solve the problem of outlier features that are extreme values, reducing the precision for all other values.


# 8-bit Quantization with LLM.int8()
LLM.int8() from [[Tim Dettmers]] (2022) is a solution to the outlier problem, relying on a vector-wise [[Absolute Maximum Quantization|Absmax Quantization]] scheme, and introducing mixed-precision quantization.
	- This means that outlier features are processed in a FP16 format, to retain their precision, while the other values are processed in an INT8 format.
	- Because outliers represent about 0.1% of values, this effectively reduces the memory footprint of the LLM by almost 2x.

![[Pasted image 20240627153728.png]]

Works by conducting MatMuls in three key steps:
1. Extract columns from the input hidden states X that contain outlier features, using a custom threshold.
2. Perform the MatMul of the outliers using FP16, and the non-outliers using IN8 with vector-wise quantization (row-wise for hidden state X, and column-wise for weight matrix W).
3. *Dequantize* the *non-outlier* results (from [[INT8]] to [[float16|FP16]]) and add them to the outlier results to get the full result in FP16.

![[Pasted image 20240627153922.png]]

This approach is necessary because 8-bit precision is limited and can lead to substantial errors when quantizing vectors with outliers. These errors tend to amplify as they propagate through multiple layers.

We can easily use this technique with [[bitsandbytes]], we just need to specify `load_in_8bit=True` when loading the model in HF:
```python
device = torch.device("cuda")
model_int8 = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto, load_in_8bit=True))

model_int8.get_memory_footprint()
Model size: 176,527,896 bytes  # This is 3x smaller than the original 487MB
```

The authors of LLM.int8() show that the performance degradation is so low it’s negligible (<1%). However, it has an additional cost in terms of computation: LLM.int8() is roughly about 20% slower for large models.








