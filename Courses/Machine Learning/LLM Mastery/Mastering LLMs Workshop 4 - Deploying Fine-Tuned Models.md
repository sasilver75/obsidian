Speakers:
- Travis Addair
- Charles Frye (Modal)
- Joe Hoover

Today is one of the deepest and most complex topics in the workshop.

Agenda:
- Serving Overview: Dan Becker
- Deployment Patterns: Hamel Hussain
- NVIDIA Inference Stack: Joe Hoover
- Lessons from Building a Serverless Platform: Travis Addair
- Batch vs Real Time and Modal: Charles Frye

---

## Serving Overview (Dan Becker)
- There are full fine-tuner where you train all the weights in the model, but let's first focus on deploying models finetuned with [[Low-Rank Adaptation|LoRA]] and similar techniques.
- ![[Pasted image 20240612151624.png|300]]
- If you imagine that for any given matrix, if the input representation was 4000 dimensions and the output was 4000 dimensions, then the matrix taking you from input to output would be 16M (4000\*4000). If you instead train with a low-rank adaptation, we train two lower-rank matrices (4000x16 and 16x4000 = 128k) and save a lot of weights to tune.

![[Pasted image 20240612152326.png|300]]
After you've trained your LoRA weights, you have a decision:
1. Keep the LoRA weights in a separate file, and then deploy with that.
2. Merge the LoRA weights into the original weights (by first multiplying the two LoRA matrices and then adding), so that you have just a single file.

![[Pasted image 20240612152449.png|300]]

Performance vs Costs:
- GPU Speed: If you're using your GPU at 100% utilization, then using a more powerful GPU might be more of a pure win... But it's more expensive.
- Model size: Running larger models is higher performance, but they're slower and/or you need more GPUs the run it at the same throughput.
- Engineering efficiency: Mostly at the platform level.
- Cold starts v Idle time: =="Loading a model onto a GPU" takes anywhere from 30s to 8 minutes 😱 These cold starts can be a show-stopper!==
	- Potential win from hop-swapping: Having a base model where you swap out smaller LoRA adapters can make it more reasonable to serve many users.

![[Pasted image 20240612153031.png]]
- Alt Text descriptions has a queue of images coming in every day, and then every day processing them as a batch. Humans review these during the day. Once we process the batch, we scale the GPU use down to 0, and we don't have any more requests until the next morning.
- Extracting chemical properties projects... again, we do this in batch every morning.
- Editing articles project, again -- we have a queue of documents, we run the queue once a day, and humans review the edits over the course of the day.
- Text-to-SQL -- This one used OpenAI APIs, so we didn't have to think deeply about how that LLM got served.

![[Pasted image 20240612153251.png]]
- Above: Top is LoRA directory after training. 168MB of weights. We have a step where we merge that into the base model, which creates the directory at the bottom. We have 4 binaries (it's sharded, because they're big files), and there are 16GB of weights.
	- Loading 16GB into your VRAM is a lot.

After that, we can then deploy our model files to HF Hub, if we want to serve our models using [[HuggingFace]] Inference Endpoints (There are many other options for how to serve models).

![[Pasted image 20240612153654.png]]

The steps here are to create a repository; We create a HF repo called conference-demo, copy the weight files into the new directory, then we commit the files the push them.

## Deployment Patterns: Hamel Hussain
- Let's talk a bit about model deployment
![[Pasted image 20240612154203.png]]
- Above:
	- Speed: Is it okay if results are slow?
	- Scale: Do we have many requests at the sam time?
	- Pace of Improvement: Do we need to constantly improve our model?
	- Real-time inputs: Do we have streaming inputs?
	- Reliability requirement: Will it be catastrophic if your model server goes down?
	- Complexity of your model: What are the resource requirements of your model?
The more things you have on the right side of the column, the more complicated it gets. If you stack too many things, then sometimes you need to build custom stuff. The off-the-shelf tools are always getting better, though.

![[Pasted image 20240612154739.png]]
Turns out you can also use Modal for inference in many ways too.
NVIDIA stack is very performant but very difficult to use; lots of knobs to tune.
You want to pay attention to things like Quanitaztion, which can make things much faster, with some performance caveats (use Evals to make sure it still meets your requirements!)

![[Pasted image 20240612155138.png]]
[[vLLM]] is the most ergonomic; I recommend this unless you need the highest performance!

![[Pasted image 20240612155416.png]]
[[Replicate]]: It can be nice to have a playground for your business people to play with. Replicate can provide a good version of this, if you're interested! The predictions have permalinks, so you can play around and send situations where something is a little weird. This is a really useful feature! Also lets you save examples, export the code to call this as (eg) Python, etc.

`cog` is a wrapper around Docker. It helps avoid a lot of CUDA hiccups. Even when using the right NVIDIA base image, you constantly have to fight CUDA problems. It's basically a Dockerfile with a  bundled webserver in it; you specify a `predict.py` file. It might have a prompt template, a Predictor(basePredictor) class, which has a setup fn (loads model with eg vLLM code), and a predict fn. You can then use `cog` to push your model to (eg) Replicate.

## NVIDIA Inference Stack: Joe Hoover (MLE@Replicate)
- Serving is really complicated! Let's talk through some of the things that have hurt Joe the most, and then talk about Replicate, and how it solves some of these problems around serving language models.
- Deploying LMs is hard because performance is multidimensional and zero-sum. IF you don't care about performance or cost and just want a simple deployment, then maybe deploying LMs isn't too hard. Lots of serving frameworks and platforms exist... but if you DO care about these things, everything becomes very complicated.
1. Performance is multi-dimensional
	- Optimizing certain dimensions of performance results in penalization in other dimensions of performance. There will be tradeoffs! ==Think very clearly and carefully about your success criteria; what's important to your users and your platform?==
2. Technology never stops evolving
	- Proliferation of serving frameworks and tools in the last two years. Part of the problem is now "keeping up."
	- You should prioritize modularity and minimize the cost of experimentation!

LLM Performance Optimization
- What makes LLMs slow?
	1. ==Memory bandwidth bottlenecks==
		- Transformers are memory-intensive creatures.
		- Operations requires transferring data from slow device memory into smaller, faster memory caches. This is a performance bottleneck!
		- ==We fix this by **transferring less data==!**
			- We use smarter kernels (e.g. fusion, flash attention, paged attention). Kernels are just functions that run on GPUs. You might have a softmax, layernorm, attention kernel, etc.
				- Fusion is when we combine kernels in an intelligent way!
				- [[FlashAttention]] does Kernel fusion and tries to minimize data transfer.
				- [[PagedAttention]] is similar; more efficient data storage and data transfer.
			- Use smaller data (e.g. quantization, speculative decoding)
				- Quantize your model; if you weights are smaller, you don't need to transfer as much data.
				- [[Speculative Decoding]] also fits in this regime.
	2. ==Software overhead==
		- Every operation (eg attention, layer norm) requires launching a kernel. This requires communication between CPU and GPU. This is another bottleneck.
		- ==We fix this by **minimizing overhead**.==
			- Using smarter kernels (eg kernel fusion -> fewer kernels, fewer launches)
			- Using ==CUDA Graphs==
				- As you do a forward pass through a model, there are a series of operators/kernels that need to be scheduled/launched/executed. A CUDA graph is a trace across all of those operations.
				- After you assemble a CUDA graph, you can launch all of your kernels as a single unit.
![[Pasted image 20240612161712.png|300]]
All of the Transformer optimization efforts usually boil down to mitigating one or both of these bottlnecks.
![[Pasted image 20240612162739.png]]
Above: [[KV Cache]]. An intersting one is just making your inputs shorter (query/prompt rephrasing/shortening?) or our outputs shorter (brevity)

![[Pasted image 20240612162917.png]]
Above: [[Continuous Batching]]
- Before continuous batching, you have to assemble each item in the batch... people would wait for a small period of time... pause... wait for requests... assemble them into a batch.. and run inference over a batch. This was a terrible way to serve LMs! 
	- You also have to wait until all the generations (many of which can be nondeterministically long) before waiting for responses.
- Continuous Batching solved this problem (and the one of injecting new requests into a batch during inference). Instead of thinking of inference as operating over steps, we think of it as operating over steps. We run multiple inference steps, each producing a single token. If we orchestrate batching from this perspective, we can introduce new items to our batch, and decode the next token. If a generation completes, we can just pull it out of our batch and return it to the user.
	- This adds some complexity in situations where you care a lot about performance.
	- A consequence: You can end up with dynamic batches sizes of (1, 15, 10, 2).
- To be clear: One of the "downsides" of continuous batching is that the batch size is dynamic, so if a flood of requests come in, the batch size will increase, and per-user performance will drop.


![[Pasted image 20240612165146.png]]
All are really cool and have different affordances, many doing things like continuous batching, share kernels, etc. All use many of the same optimization techniques.

Where things get tricky is when you care about performance.
There are 3 ways to think about performance:
![[Pasted image 20240612165258.png]]
This gets tricky when thinking about these in the context of continuous batching. Specifically the relationship between TPS and Single-Stream TPS.
Above: ==Single-Stream Tokens per Second==

![[Pasted image 20240612165412.png]]
A lot of products make claims of 150Tok/Sec or something, but then you make requests against the service, and you get 40Tok/sec. That's because the servers are quoting maximum single stream TPS @ a batch size of 1. But recall that batch size is changing under continuous batching!

![[Pasted image 20240612165921.png]]
Above: If you're generating Synthetic Data (something that is done "offline," and doesn't have per-request latency requirements), you want to increase your batch size!
- Inversely, if you want the fastest single-stream TPS, then you should literally run at a batch size of 1.



## Lessons from Building a Serverless Platform: Travis Addair
- Optimizing Inference for Fine-Tuned LLMS: Lessons learned @ Predibase building a platform for training and serving Fine-tuned LLMs
- Predibase founded out of Uber's Michaelangelo ML platform team; Travis is also a maintainer of Horovod, Ludwig, and [[LoRAX]].
- Thesis: Bigger isn't always better. General intelligence in great, but you don't need your Point of Sale system to be able to recite French Pottery (and you're paying for that extra ability in one way or another, whether in dollars or latency).
- So you often start with GPT-4, but then move to smaller, fine-tuned, open-source models. But fine-tuning isn't free!

How do we deploy fine-tuned LLMs (the old fashioned way?)
- Have a Kubernetes Pod. A request queue comes in, and then you have your fine-tuned model weights. If you're doing PEFT, these weights only account for 1-5% of the parameters of the model... but when you're deploying all these things, you're deploying them over and over again with the same base model parameters replicated over and over again.![[Pasted image 20240612173031.png|200]]
- What if we just took these base model parameters and tried to serve the multiple finetuned model parameters (red, blue) in the same model deployment? This led to [[LoRAX]]
- ![[Pasted image 20240612173111.png|300]]
- LoRAX supports serverless inference on Predibase; we take all these different user requests coming in for various fine-tuned models, and serve them all from the same deployment (assuming that they use the same base model).
- We batch together requests for various adapters and execute them at the same time. ![[Pasted image 20240612173200.png|350]]
- We can also lower this down to the CUDA level and use indirection through pointers, different tiling strategies, and do it more efficiently.
- ![[Pasted image 20240612173318.png]]

What about adapter merging?
![[Pasted image 20240612173532.png]]
- You get good performance because you don't have to pay overhead for processing additional layers at runtime.
- Downsides though: If you need to serve multiple finetunes... It turns out that that the idea of merging weights back into the base model gets tricky in a world where you finetuned the base model using [[Quantized Low-Rank Adaptation|QLoRA]]. Those newly trained weights are intrinsically tied to the quantized model.

![[Pasted image 20240612173848.png]]
If you finetune a model using QLoRA, how are you going to serve the model? Maybe you'll think that you'll just serve it using full-precision of half-precision FP16, because you don't want to pay the cost of dequantizing... but the activations in red are going to be different, and may change or damage performance!

So what if we just do QLoRA at inference time? Unfortunately performance is significantly worse using quantized models.
![[Pasted image 20240612173939.png|450]]
If you can get away with it, you'd rather not serve with QLoRA.

So it seems like we have a dilemma!
![[Pasted image 20240612174023.png]]

Solution: *Dequantize* the QLoRA weights
![[Pasted image 20240612174320.png]]
You take the original weights of the model in FP16, quantize them using Bits n Bytes, and then just reverse the quantization, and the result is a set of fp16 weights that's numerically identical to the quantized weights. Now you have none of the performance degredation that we had before!
- ((I wonder if his first image was fucked up))

![[Pasted image 20240612174541.png]]

![[Pasted image 20240612174657.png]]
Ballpark: At least 1.5x the model weights is needed to serve the model. You also need to store activations, adapters, and KV Cache.

## Batch vs Real Time and Modal: Charles Frye - Deploying LLM Services on Modal
- When a software system is slow, that could mean:
	- It took a long time for a big thing to get completed
		- System doesn't have sufficient throughput
	- It look a longer than expected time for a small thing to get completed
		- System doesn't have sufficient latency
- When we optimize these two things, we can deploy resources to achieve different services levels on throughput and latency, and *cost* becomes the hidden third feature determining both throughput and latency.
- Throughput
	- Batch: Thinking about batch-oriented LM inference. Refreshing recommendations for users, evaluation in CI/CD.
	- Real Time: Chatbots, copilots, audio/video, guardrails... where we have tight constraints on throughput (and latency, in real-time cases)
	- Cost: Consumer-facing, large-scale scenarios are those where it's most difficult to exchange throughput for latency.
	- With Throughput, you're generally bottlenecked by upstream and downstream services. If you're generating 10,000 tokens a second, and your logging server can't handle it, then you'll be bottlenecked.
- Latency
	- Human perception of latency is what matters, here. You can get around latency constraints by doing tricks like showing intermediate results, or showing a draft that you later update. Here, latency is going to be on the order of a couple hundred milliseconds for the *entire system*. As long as it fits in some low hundred milliseconds, the *human* becomes the bottleneck.

==Latency Lags Throughput==

==It's much easier to improve bandwidth than it is to improve latency.==

> "There's an old network saying: Bandwidth problems can be cured with money. Latency problems are harder because the speed of light is fixed; you can't bribe God."

You can't bribe physics, but you can spend more money on things to create better bandwidth.

GPUs are inherently throughput-oriented designs, whereas CPUs are inherently latency-oriented (having a lot of space dedicate to caching, and less to ALUs for throughput).
![[Pasted image 20240612180822.png]]
GPUs on the other hand are throughput-oriented processors where most of the space is given over to processing, and a small amount for high-speed caching. GPUs have substantially-higher memory throughput than CPUs.

This is a general phenomenon that's very relevant in GPUs, and is very punishing in LLM inference in particular

- Want more throughput? Just increase batch size!
	- Penalties to latency, but ~linear scaling in throughput as you scale batch size, up to compute-boundedness, and which point just get more GPUs and keep increasing throughput until you start to run out of powerplants.
- Want shorter latency? Just go die.
	- Quantize (also improves throughput)
	- Distill models to smaller sizes, or truncate them (also improves throughput)
	- Buy more expensive hardware (also improves throughput)
- Want REALLY short latency?
	- Run the entire system on Cache RAM/SRAM (this is not the DRAM ram you're used to thinking about; this is the L1/L2/L3 cache in your CPU). Build 80GB of that and run the model off that.
	- If you want to do this, be [[Groq]], which basically build their LPU off this.
	- Penalties here to throughput per *dollar*.

[[Modal]]
- Throughput
	- Easy to scale out to hundreds of A100s or thousands of A10Gs, flexibly.
- Latency
	- Challenging to build latency and cost for models > 13B (for now). 
- Cost
	- $1.10/hr per A10G (cheaper than AWS)
	- $7.65/hr per H100

It's hard to achieve high GPU utilization; Average GPU utilization at peak is only 60%, but most providers charge you for 100%!