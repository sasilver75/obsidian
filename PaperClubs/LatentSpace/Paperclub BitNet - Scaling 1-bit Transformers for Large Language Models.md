
October 17, 2023
Link: https://arxiv.org/abs/2310.11453

---

The increasing size of LLMs has posed challenges for deployment and has raised ethical concerns about environmental impact

[[BitNet]] is introduced as a scalable and stable 1-bit Transformer architecture for LLMs!
- Specifically they have  ==BitLinear layer that replaces the nn.Linear== to train 1-bit weights from scratch.
- Achieves competitive performance while reducing memory footprint and energy consumption.
	- Exhibits a ==scaling law== akin to full-precision Transformers, suggesting its potential for effective scaling.

![[Pasted image 20240410113614.png]]
Above:
- See energy cost reduction increasing vaguely linearly as you increase model size
- See loss decreasing at roughly the same rate as a larger-precision FP16 model as you increase model size (re: scaling laws)
- See that, unlike other quantization methods, BitNet lets you not only do inference at a low precision, but do training too!


Most existing quantization approaches for LLMs are post-training; they are simple and easy to apply since they don't require any changes to the training pipeline or retraining the model.
- ==These existing approaches  (that train at full precision and then lobotomize themselves) result in a more significant loss of accuracy==, since the model wasn't optimized for the quantized representation while training.

Quantization-Aware training *does exist* - it typically results in better acuracy, since the model is trained to account for the reduced precision from the beginning... The challenge here mainly lies in optimization -- the model becomes difficult to converge as the precision goes lower.

In this work, we focus on ==binarization== of the model (1 bit) -- the extreme case of quantization.
- ==BitNet employs low-precision binary weights and quantized activations, while maintaining high-precision for both the optimizer states and gradients during training==.

The implementation of BitNet is ==simple== and only involves replacing the linear projections (nn.Linear in PyTorch) in the Transformer; ==it complements other acceleration methods for LLMs, such as [[PagedAttention]], [[FlashAttention]], and [[Speculative Decoding]]==

![[Pasted image 20240410114141.png]]
Above:
- It uses the same layout as transformers, in terms of there being stacked blocks of self-attention followed by FFNNs
- Within the FFNNs, we simply replace the nn.Linear layers that sandwich the nonlinear activation function with BitLinear layers!
	- This employs binarized (1-bit) model weights.

They leave the other components as high-precision (8-bit in their experiments) -- they summarized the reasons as follows:
- The [[Residual Connection]] and the [[Layer Normalization|LayerNorm]] contribute negligible computations costs to LLMs
- The computation cost of Attention/QKV transformation is much smaller than the parametric projection as the model grows larger.
	- ((I thought I remember reading an example where something like 30% of the model weights were in attention blocks, vs FFNN blocks? That's not nothing! I thought in the rag space I've been hearing about using binary feature spaces to retrieve information; that seems very similar to the attention computation, in a way? Is there room for the attention computation to be binarized?))

## BitLinear
- We binarize the weights to either +1 or -1 using {technique}
- We further quantize the activations to be b-bit precision using {technique
- ...

One essential technique to scale LLMs is [[Model Parallelism]], which partitions the matrix multiplications onto multiple devices; a prerequisite for this is that the tensors are *independent* along the partition dimension. The above process breaks this assumption, so they introduce an all-reduce operation for each introduced parameter that somehow fixes this.

# Model Training
- To train our 1-bit model, we employ the straight-through estimator (STE) to approximate the gradient during backpropagation!
	- This method is able to bypass the nondifferentiable functions (such as the Sign and Clip functions) during the backwards pass, letting the gradient flow through the network.

Mixed precision training:
- The weights and activations are quantized to low precision, but the gradients and optimizer states are stored in high precision to ensure training stability and accuracy.
- We maintain a latent weight in a high-precision format for the learnable parameters to accumulate parameter updates... the latent weights are binarized on the fly during the forward pass and not used for the inference process (==?==)


Large learning ratre
- One challenge is that a small update on the latent weights often makes no different in 1-bit weights; this results in a biased gradient and update which are estimated based on the 1-bit weights.
- The problem is even worse at the beginning of training, where models are supposed to converge as fast as possible. The authors conclude that raising the learning rate is the simplest way to accelerate the optimization -- we see that BitNet benefits from a large LR in terms of convergence, while FP16 Transformers would diverge at the beginning of training with the same LR.

# Comparison with FP16 Transformers
- To study the scaling laws of the binarized Transformer, we fix the number of training tokens and vary the model size, noting the loss achieved at each model size.
	- We then fit a line and see that it looks very similar to the FP16 line.

![[Pasted image 20240410115215.png]]
Above: Note that given a fixed compute budget, BitNet achieves a significantly better loss -- or, the inference cost is much smaller to get the same performance as the FP16 models. BitNet similarly has a much higher scaling efficiency when it comes to loss a a function of energy consumption.
## Results on downstream tasks
- In addition to the loss being slightly higher loss per parameter count, the capacity is more difficult to predict due to emergent nature of NN models.
- We evaluate on [[HellaSWAG]], Winogrande, [[Winograd]], and Storycloze

((==I don't see the results on each of these, specifically?==))
![[Pasted image 20240410115540.png]]

A major challenge for training low-bi Transformers is  the stability in optimization
- We perform comparative stability tests and show that BitNet can converge with large LRs while FP16 Transformers cannot, demonstrating better training stability of BitNet.


# Comparison with Post-training Quantization
- We compare with SoTA quantization methods, including Absmax, SmoothQuant, GPTQ, and QuIP -- these methods are post-training quantization over an FP16 Transformer model, following the same training setting and data s BitNEt.

results:
![[Pasted image 20240410115853.png]]
Above:
- We evaluate the method across several weight bit levels, spanning from 16 down to 1.
- Besides the zero-shot accuracy on downstream tasks, the evaluation metrics include LMs perplexity, etc.
- The results demonstrate the effectiveness of BitNet in achieving competitive performance levels compared to the baseline approaches


The results demonstrate the effectiveness of BitNet in achieving competitive performance levels compared to baseline approaches, particularly for lower bit levels.

----

