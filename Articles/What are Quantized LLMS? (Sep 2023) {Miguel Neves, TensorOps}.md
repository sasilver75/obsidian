https://www.tensorops.ai/post/what-are-quantized-llms

---
Model quantization is a technique used to reduce the size of neural networks by reducing the precision of their weights.
- It can make it possible to allow you to run your model on less powerful hardware with cheaper/faster inference while retaining nearly all of the capabilities of the model.

## What is Quantization?
- The process of converting the weights of the model from higher-precision data types to lower-precision ones.
- To store weights in NNs, we can use different types:
	- Float64
	- Float16
	- integers
	- etc.
- The datatype you use will impact both the memory size and performance of your model. The number of "digits" is referred to as the precision of the data type.

Companies have developed hardware (GPUs) and frameworks (PyTorch) to support lower-precision operations.
- Google's TPUs introduced the concept of ==[[bfloat16]]== (brain floating point 16, a special primitive data type optimized for neural networks.

![[Pasted image 20240618104932.png]]

![[Pasted image 20240618105116.png]]
Above:
- For reference, an NVIDIA A100 has 80GB of memory in its most advanced version, and you can see that LLaMA2-70B requires 138GB of memory to host it (more, if you want to do training, since you have to keep track of optimizer/forward/backward pass states) -- this would require us to have multiple A100s and additional infrastructure.
	- An *INT4* quantized version of the same LLaMA2-70B model. however, only requires 40.7GB, so it can easily fit onto one A100, reducing the cost of inference and increasing its speed.

Quantization works by reducing the number of bits required for each model weight. A typical scenario would be the reduction of the weights from FP16 (16 bit Floating-Point) to INT4 (4-bit Integer), which allows the model to run on cheaper hardware and/or with higher speed (albeit at some small quality impact).

### Two Types of LLM Quantization
- ==Post-Training Quantization== (PTQ): Converting the weights of an already-trained model to a lower-precision, without any retraining. Though it's straightforward to implement, PTQ might degrade the model's performance slightly due to loss of precision in the value of the weights.
- ==[[Quantization-Aware Training]]== (QAT): Unlike PTQ, QAT integrates the weight conversion process during the training stage. This often results in superior model performance, but it's more computationally demanding.

### Larger Quantized Models vs Smaller, Non-Quantized models
- Which of these two options should we prefer, with the assumption that they both take about the same amount of VRAM?
- [Research from Meta](https://arxiv.org/abs/2305.17888)  seems to indicate that in many cases the quantized model demonstrates superior performance, and also reduced latency and enhanced throughput. Larger models exhibit smaller quality losses when quantized, so the advantage is even more pronounced.

You can find many models always quantized on the HuggingFace Hub using methods like:
- [[GPTQ]]
	- This method mainly focuses on GPU execution
- [[NF4]] (Normal Float 4-bit)
	- Implemented on the [[bitsandbytes]] library, it works closely with the HuggingFace `transformers`  library. It's primarily used by [[Quantized Low-Rank Adaptation|QLoRA]] methods, and loads models in 4-bit precision for fine-tuning.
- [[GGML]]
	- This C library works closely with the llama.cpp library, and features a unique binary format for LLMs, allowing for fast loading and ease of reading. Notably, the recent shift to [[GGUF]] format ensures future extensibility and compatibility.
A substantial number of these models have been quantized by TheBloke, an influential/respected figure n the LLM community.

Post-training quantization model compression can be time-consuming, even though it doesn't involve ; a 175GB model demands at least 4 GPU-hours.

## GPTQ: A revolution in model quantization
- The name of GPTQ merges the name of the GPT model family with post-training quantization (PTQ)
- Benefits include:
	- Scalability (Can compress 175B param models in about 4 hours)
	- Inference Speed: GPTQ models offer 3-4x speedups on NVIDIA high performance GPUs.
- ==GPTQ can only quantize models into INT-based data types, being most commonly used to convert to 4INT,== but can also do 3, 4, 8-bit representations.

## NF4 (NormalFloat 4-bit ) and bitsandbytes
- The NormalFloat (NF) data type is an enhancement of the ==Quantile Quantization== technique, which has shown better results than both 4-bit Integers and 4-bit Floats.
- Can also be coupled with ==Double-Quantization (DQ)== for high compression while maintaining performance.
- NF4 and Double-Quantization can be leveraged using the [[bitsandbytes]] library, which is integrated in the `transformers` library. 
## GGML and llama.cpp: Allowing inference on CPU
- [[GGML]] is a C library for ML, where the GG refers to the initials of its originator, [[Georgei Gerganov]]. It was later improved on with [[GGUF]]
- The library's distinctive edge lies in its proprietary binary format that offered a way to distribute models.
- GGML was meticulously crafted to work seamlessly with the [[llama.cpp]] library, whose main goal is to allow the use of LLaMA models using 4-bit  integer quantization on a MacBook.
	- Though nowadays, it also allows to offload layers onto a GPU.


