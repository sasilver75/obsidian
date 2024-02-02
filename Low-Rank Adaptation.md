---
aliases:
  - LoRA
---
- ==LoRA can reduce the number of parameters by 10,000 times and the GPU memory requirements by over 3 times.==


- ==By introducing pairs of rank-decomposition weight matrices (known as update matrices) to the existing weights, LoRA focuses solely on training these new added weights.==
- This approach offers several ==advantages==:
	1. ==Preservation of pretrained Weights==: LoRA maintains the frozen state of previously-trained weights, minimizing the risk of catastrophic forgetting. This ensures that the model retrains its existing knowledge while adapting to new data.
	2. ==Portability of trained weights==: The rank-decomposition matrices used in LoRA have significantly fewer parameters compared to the original model. This allows the trained LoRA weights to be *easily transferred* and utilized in other context, making them *highly portable*.
	3. ==Integration with Attention Layer==: LoRA matrices are typically incorporated into the attention layers of the original model. Additionally, the adaptation scale parameter allows control over the extent to which the model adjusts to new training data.
	4. Memory efficiency: LoRA's improved memory efficiency opens up the possibility of running fine-tune tasks on less than 3x the required compute for a native fine-tune!


Related: [[Quantized Low-Rank Adaptation]]