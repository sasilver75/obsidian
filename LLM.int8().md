References:
- Article: [Introduction to Weight Quantization from Maximme Labonne](https://towardsdatascience.com/introduction-to-weight-quantization-2494701b9c0c)


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