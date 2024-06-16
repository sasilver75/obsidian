# Topic: Fine-tuning and Instruction Tuning
https://www.youtube.com/watch?v=KLJ3EEo8aPU&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=8

---

Full Fine-tuning
- We simply continue training our model on whatever task that we want
- This can take lots of memory, and can even be relatively unstable to other alternatives!
	- For example, a 65B model (largest LLaMA 1) with 6-bit precision takes much more memory than you'd expect!
![[Pasted image 20240615171306.png]]
Forward/Backward pass depend on how many tokens you have in your sequence, etc.
Overall, this takes ~1000GB of memory, which isn't great!

Luckily, there have been some advances since then.

![[Pasted image 20240615171533.png]]
See that our best available GPUs right now are 80GB; the B200 GPU will pack up to 192GB, a 2.4x increase over H100s.
So it goes without saying that you've got yourself a distributed systems problem, even for finetuning a relatively small 70B model.

So how can we overcome this limitation of GPU memory?

MultiGPU Training!
- We can just throw more hardware at the models, and distribute the models over multiple GPUs.
- ![[Pasted image 20240615172228.png]]
- The most canonical version of this is called DeepSpeed Zero
	- Works by partitioning optimization over different devices.
	- There are different stages of DeepSpeed Zero;
		- The first is this Baseline Row; we hold all of these on each GPU
		- The second (stage one) is that we partition the optimizer state across GPUs. This is a big win, because the optimizer state takes up so much memory.
		- Stage 2 involves partitioning the gradients, which gets tricky because you have to move things around between devices a lot
		- Stage 3 involves doign this with parameters to; this gets very expensive in terms of moving things around so you can calculate your gradients appropriately.
	- By default, Stage 1 or Stage 2 are what the speaker recommends.

"Axolotl, a lot of people are using... TRL we might have a recitation on later."

![[Pasted image 20240615172425.png]]
The other option besides parallelizing a full finetune across hardware is just training *some* of the parameters in the model. This is [[Parameter-Efficient Fine-Tuning]]

The first is something like [[Prefix Tuning]], which is like a bridge between PEFT and prompting; it tuns one prefix for each of the layers.
![[Pasted image 20240615172517.png]]

The next is ==Adapters== (Houlsby et al. 2019). 
![[Pasted image 20240615172533.png]]
We have our standard Transformer architecture... we add some additional linear layers into transformer blocks and train those parameters only, leaving the rest frozen. These new layers are called [[Adapter]]s.
- Q: Why do we make it smaller then larger?
- A: That's 'a way to reduce the parameter count'
NOTE: ==Even though we're only finetuning the adapters, we still need to store the gradients for other layers (specifically those that are on the path to the gradients of parameters that we want to optimize)==

![[Pasted image 20240615173555.png]]
Here's another interesting technique called [[Adapter Fusion]]; the idea is to learn an adapter for various types of tasks, and the combine them together
- We have multiple adapters, and then adapter fusion, where adapter fusion is really just *attention over adapters*. These adapters are trained separately on task-specific data.
- In a way, this is kind of like a version of a [[Mixture-of-Experts]] model.


![[Pasted image 20240615173423.png]]
- LoRA is very popular; it works similarly to adapters, but it has an important implementation difference; in contrast to Adapters, which have nonlinear layers, LoRA has no nonlinearity; basically we take a downscale matrix and an upscale matrix, and do a linear transform with them.
- In [[Low-Rank Adaptation|LoRA]], we initialize our upscaling matrix to zeroes ("so if you don't do anything it will stay the same")

![[Pasted image 20240615173909.png]]
[[Quantized Low-Rank Adaptation|QLoRA]] combines [[Quantization]] with [[Low-Rank Adaptation|LoRA]]
- There are ways to compress the model down to not be in 16 bits, but be in 4 bits! This makes the model very compact.
- Because we compressed the model down, the parameters are small. We have a very compact LoRA layer (which doesn't take much memory itself), which lets us train a model on commodity hardware (eg 48GB GPU). It also has paging to page things from CPU to GPU memory and make it more efficient.

Definitely if you want to train a large model on limited hardware, he recommend QLoRA; else, you should probably be using LoRA.

![[Pasted image 20240615174231.png]]

![[Pasted image 20240615174249.png]]
(This is a paper from Graham Neubig)
![[Pasted image 20240615174343.png]]
Basically, we find that things like Adapters, LoRA, and Prefix Tuning are actually very similar to eachother, with the difference being
- Where do you get the original representation that you're feeding in?
	- Adapters get it from after the module that you're adapting.
	- Prefix, LoRA get it from before.
- They show that the understanding can even lead to new variants that can be more effective than the original variants
	- Parallel Adapter
	- Scaled Adapter

So which PEFT method do we choose?
- LoRA and BitFit don't change the model architecture, which is nice and conveniently easy.
- For accuracy, it doesn't really matter that much, especially for simpler tasks -- but for more complex tasks (or small parameter budget), the authors found [[Prefix Tuning]] to do a good job. For More complex tasks and larger budgets, [[Adapter]]s or Mix-and-Match methods worked the best for them.


----

## NLP Tasks

![[Pasted image 20240615175015.png|450]]
Open-Book is not Open-Domain

![[Pasted image 20240615175139.png|450]]
Closed-Book is not Closed-Domain

![[Pasted image 20240615175206.png|450]]

![[Pasted image 20240615175312.png]]
[[Summarization]]: Single-document works pretty good out of the box; pretty close to perfect in English, though multilingual is interesting
Multi-document summarization is very much not solved -- when we want to boil down multiple documents into a coherent summarization.


[[Information Extraction]]
![[Pasted image 20240615175908.png]]

[[Machine Translation]]
![[Pasted image 20240615180041.png|300]]
Converting from one language to another; For both translation and summarization, evaluation is tricky, because it's vaguely open-ended; we usually assess similarity to some gold reference using some lexical method like [[BLEU]] score, or NN methods.

Then there are some "general purpose" benchmarks ((Though I wouldn't say that they're about "general purpose use" of the models; ie I don't ask my models about MMLU questions))
![[Pasted image 20240615180648.png]]
This includes things like [[BIG-Bench]] and [[Massive Multi-Task Language Understanding|MMLU]]


They do a little talking on instruction tuning (mentioning a few papers), and then talk a tiny bit about dataset generation (self-instruct, ORCA, Evol-Instruct).
