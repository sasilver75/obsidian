Feb 27, 2024
Link: https://arxiv.org/abs/2402.17764

----

This follows the paper [[BitNet - Scaling 1-bit Transformers for Large Language Models]]

Recent research is paving the way for a new era of 1-bit LLMs. In this work, we introduce a 1-bit LLM variant, namely ==BitNet b1.58==, in which ==every single parameter of the LLM is a== ==ternary== {-1, 0, 1}.

The performance matches the full-precision (i.e. FP16) Transformer LLM with the same model size and training tokens both in terms of perplexity and end-task performance, wile being significantly most cost-effective in terms of latency/memory/throughput/energy consumption.

The 1.58 bit LLMs define a new scaling law and recipe for training new generations of LLMs that are both high-performance and cost-effective.

Models are getting larger and more expensive to train.
One approach to address these challenges is to use post-training quantization to create low-bit models for inference -- the trend is mostly moving frmo 16 bits to 4-bits.

Recent work on 1-bit model architectures like [[BitNet]] promise to reduce cost of LLMs while maintaining their performance.
The major computation cost comes from the floating-point addition and multiplication operations. 
- In contrast, the ==matrix multiplication in BitNet only involves integer addition, which saves orders of energy costs for LLMs!==

In addition to computation, ==the process of transferring model parameters from DRAM to the memory of an on-chip accelerator (eg SRAM)== can be expensive during inference.
- ==1-bit LLMs have a much lower memory footprint== from a capacity and bandwidth standpoint, reducing the cost/time of loading weights from DRAM, leading to faster and more efficient inference.

We introduce a significant 1-bit LLM variant called ==BitNet b1.58==, where every parameter is Ternary, taking on values of {-1, 0, 1} -- ==they added an additional value of 0 to the original 1-bit BitNet, resulting in 1.58 bits in the binary system.==
((Aside: 1.58 = $log_2(3)$      This means that we can fit floor(16/1.58)=10 of these numbers onto 16-bits, because each number takes up 1.58 bits.))
- BitNet b1.58 offers two additional advantages
	- ==Its modeling capability is stronger due to the explicit support for feature filtering, made possible by the inclusion of 0 in the model weights== (which can significantly improve the performance of 1-bit LLMs)
	- ==BitNet b1.58 can match full-precision (FP16) baselines both in terms of perplexity and end-task performance, starting from a 3b Size, when using the same configuration.==
		- ((THIS IS BONKERS? Meaning at the same parameter count, we don't even really need the full expressivity of FP16? This sound almost too good to be true, right?))


# BitNet b1.58

### Performance
![[Pasted image 20240410132753.png]]
Above:
- Note that we're comparing to LLaMA, not LLaMA 2. This might be a fine comparison thought, because LLaMA was just a pretrained model without additional tuning, whereas LLaMA2 is an SFT/RLHF'd model.
- The above table is a summarization of perplexity and cost -- it shows that BitNEt starts to match full precision LLaMA LLM at 3B model size, in terms of perplexity.
- They compared BitNet  b1.58 to FP16 LLaMA LLM in various sizes
	- We pretrained the models on [[RedPajama]] for 100B tokens, and evaluated the zero-shot performance on a range of language tasks, including:
		1. [[Abstraction and Reasoning Corpus|ARC]]-Easy
		2. ARC-Challenge
		3. [[HellaSWAG]]
		4. [[Winogrande]]
		5. PIQA
		6. OpenbookQA
		7. BoolQ
	- Also evaluated perplexity on
		- WikiText2
		- [[C4]]


The authors note that they use many LLaMA-like components in their design:
- [[RMSNorm]]
- [[SwiGLU]]
- [[Rotary Positional Embedding]] (RoPE)
- Removes all biases from the model

In this way, they hope that BitNet b1.58 can be integrated into popular open-source software (eg HuggingFace, vLLM, llama.cpp)

### Memory
![[Pasted image 20240410133424.png]]
Above:
- Showing latency and memory improvements of BitNet b1.58 versus LLaMA -- see that the number grows as you scale the model; this is because the time cost for nn.Linear grows with model size; the memory consumption follows a similar trend.
### Throughput
Above:
- Table shows that BitNet b1.58 70B can support up to 11 times the batch size of LLaMA LLM, resulting in an 8.9x higher throughput.

((It also uses less energy, whatever))

## New Scaling Laws
![[Pasted image 20240410134043.png]]
Above:
- ((I'm not sure exactly what "more efficient" means here... but I think they're saying a 30B 1.58 uses less memory than a 7B FP16 model?))



![[Pasted image 20240410134136.png]]
The number of tokens is a crucial factor for LLMs generally; we trained a BitNEt b1.48 model with 2T tokens following the data recipe of ==StableLM-3B, which is the SoTA open source 3B model==.
- ==Both were evaluated on a range of benchmarks, and the results were that BitNet b1.58 3B beat the SoTA!==

# Future Work

### 1-bit Mixture-of-Experts (MoE) LLMs
- MoE have proven to be a cost-effective approach for lLMs.
- While it significantly reduces the computation FLOPs, the high-memory consumption and inter-chip communication overhead limits its deployment and application.
	- These challenges can be addressed by 1.58-bit LLMs!
		- The reduced memory footprint reduces the number of devices required to deploy MoE models.
		- Significantly reduces the overhead of transferring activations across networks.

### LLMs on Edge and Mobile
- Reduce memory consumption of these trinarized LLMs allows them to be deployed on these devices, enabling a wide range of applications that weren't previously thought possible.

### New Hardware for LLMS (?)

The authors finish by calling for *new hardware and system software* specifically optimized for 1-bit LLMs, given the new *paradigm* enabled in BitNet!
((The paper club speaker thought that this was handwavey -- he'd like to hear more of the *why*))


![[Pasted image 20240410134847.png]]




