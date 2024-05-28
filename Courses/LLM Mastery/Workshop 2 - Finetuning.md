
Using Axolotl
- https://github.com/OpenAccess-AI-Collective/axolotl
	- Fantastic readme
	- Look for the >examples folder in the repo
		- Axolotl takes YAML config files, which are reasonably long -- very few people besides creators could write one of these configs without referencing YAML files. Note there's plugins for WandB
	- Once you have your config, click on the quickstart section and follow instructions
	- ![[Pasted image 20240521104149.png]]
	- This bottom step is kind of cool, it spins up a gradio thing -- you only want this to spotcheck your model.


![[Pasted image 20240521104814.png]]
Above: Alpaca template

![[Pasted image 20240521104627.png]]
(-100, 2899) tokenid is 2899
The -100 prevents it from influencing the behavior of our model; no parameter updates
It's useful to look at these as a sanity check (Esp during multiturn conversations) to make sure that we're doing what we think we are.

This is so we don't penalize the model for not being able to predict the other aspects of the template besides the model response. We just want it to learn to  predict the Response: {output} part of the template.

Axolotl docs have various different dataset formats

![[Pasted image 20240521110430.png]]
Eg: here's the sharegpt format

![[Pasted image 20240521110449.png]]
Here's one with a system, human, and gpt part of the prompts
- Both the system role and the human input are considered as "inputs", and the from:"gpt" is the output; we should only be penalizing/teaching the model to predict the output.

We often make mistakes in dataset preparation

We like to look at the data and doublecheck how axolotl is preparing the data

We can do that via 
!axolotocl.cli.preprocess command, which assembles the data in the right format...

We like to look at the data manually

** Go through Hamel's Axolotl blog post as homework! **

You can log your training runs to weights and biases
![[Pasted image 20240521111208.png]]
Just know that it's there, if you wanna look at it

He tried different parameters, varying the learning rate, etc.
- For Mistral 7B, he went to the discord and just asked: "What's the best config for finetuning mistral?"
- He tried different configuration schemes using Deepspeed Stage Zero/Three


![[Pasted image 20240521111828.png]]
- + A bunch of few shot examples of inputs (natural language), queries, and critiques/ratings
- He had his friend write critiques every day for a few weeks. He tried to get the model to agree with the critiques that his friend Phillip was writing. 

The thing that was really good about this

And then he did something like

Generate query -> Critique query -> ask first LM: Can you make the query better? -> repeat ..

From there then, you can curate your data!
- Fixing the bad data

You might want to filter the data
- When we talk about dataset curation, there are many things you can do! Filtering, you want to use both your level 1 evals (Assertions, etc), level 2 evals to do more filtering...
- And apply other filters! The model is making a certain type of mistake, let me filter that type of mistake out, and decide whether you have togo acquire more data for that type of mistake. One example of that... is that he realized there were a lot of low complexity queries and lot of very very very high complexity queries that didn't make sense
- He wrote some code to filter both of these out

Getting rid of duplicates is important too

There's a tool called **`Lilac`*** that helps you find things you might want to filter out
- Has things like duplicates, semantic and keyword search, clustering, fuzzy concept search, etc.
- The idea is to look at data, try to maximize its diversity, and filter out things that have too much duplication.



Debugging Axolotl
- https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docs/debugging.qmd
- Use the latest version of Axolotl
- Eliminate concurrency
- Use a small dataset
- Use a small model
- Minimize iteration time
- Clear caches (huge, especially if you're trying to debug something about dataset formation)



---

Zach Mueller 
Technical lead for the HuggingFace Accelerator project

![[Pasted image 20240521113022.png]]

![[Pasted image 20240521113040.png]]
If each parameter in a model is 4 bytes, and the backward pass takes ~2x the model size, and the optimizer step takes ~4x the model size (2x optimizer when it comes to ADAM)

Then it comes out to 1.6GB is needed to train on a batchsize=1 for BERT

To use mixed precision, if the model is full precision, the gradients might wind up taking less, because the gradients are in half-bit

Why does this matter?

![[Pasted image 20240521113136.png]]
In a sense, this use of lower-precision doesn't matter as much, because we rarely have 56GB or VRAM on single cards anyways

If we want to avoid things like PEFT, what do we do?

Distributed Training -- using multiple GPUs!

![[Pasted image 20240521113228.png]]
DDP
- Have a full model on every device, with data chunked between each GPU; we can process the data faster because we're sending chunks of our full batch across multiple GPUs to speed up training time.
FSDP and DS
- This is what we'll talk most about
- These are the key areas hinted at where we can split chunks of the model and optimizer states acrtoss multiple GPUS, allowing for us to use basically a larged virtual memory
	- Unlike with DDP we're stuck with 2 4090s at 24GB, so that's all we can do is use 24GB

Fully Shared Data Parallelism
- Take our model, and create shards of our model (Imagine the model split perfectly in half)
- Occasionally, Torch will need ot know what's happening with the other model chunk; this results in "communication" calls, which are time spent on our GPUs just talking to eachother trading notes on how they think the model should be, and correcting themselves.
- FSDP: Getting Parameter Specific
	- Different parameters can dictate how much memory is needed for total GPU training across multiple GPUs
	- These include how model weights are sharded, gradients, and more
	- I'll cover some more important ones needed when doing a full fine-tune of LLaMA 8B

![[Pasted image 20240521113404.png]]

![[Pasted image 20240521113519.png]]
Us telling FSDP how we want to split up things that take up VRAM
- FullShard: All things are split
- ShardGradOp: Only shard the optimizer states and gradients; the model will be split when we're not using it, and joined back togehter when we are, like in the backward pass. We're still fitting th entire model in VRAM, but reduces the training VRAM a little bit for us
- NoShard: Normal distributed data paralellism, no sharding
- HybridShard: New thing from Pytarch; Kind of like FullShard, but if we're training on multi-node (multiple computers), it keeps a copy of the entire model on each of those nodes. This lets us reduce the communications from 3 down to 2 if not 1, so our training speed is increased exponentially, depending on how long it takes for your computers to talk to eachother.


![[Pasted image 20240521113703.png]]
We know how to split the memory, but how do we split the model?
- TransformerBasedWrap: Very specific to transformers; declare the layer you want to split on (some BERT layer, or aLLaMA layer; has good defaults)
- SizeBasedWrap: Basically telling FSDP: "After X amount of paraemters, dgo ahead and split hte model" This is good because it's easy, but bad because there are speed increases you might be missing (eg having each head of a mistral model o na separate GPU, which would be great to reduce communication)


![[Pasted image 20240521113810.png]]
- Okay, I have 48GB of VRAM, and I can't fit that :( 
- Let's say we still want to do it, and now use a cloud prameter
- This lets us offload GPU memory into RAM (this will be extremely slow, going from GPU -> CPU -> RAM), but it lets us train as big a model as we have available RAM, but is VERY SLOW.
- Case: LLaMA-3-8B finetuning (tradeoff of time and $)

![[Pasted image 20240521113935.png]]
Another critical part when it comes to using FSDP is this idea of CPU/RAM-efficient loading
- Pytorch lets use a thign called device=meta, which is a skeleton of your model (weights aren't loaded, can't do computations) that we'll eventually load weights into.

![[Pasted image 20240521114121.png]]
HuggingFace Accelerate is the foundatinon of many of your favorite libraries!
- The general idea with accelerate is that it's essentially three frameworks
	- Command line interface
	- Training library: Under the hood, what does this distributed training
	- Big model inference: we don't care about this here

![[Pasted image 20240521114159.png]]

![[Pasted image 20240521114209.png]]
- Accelerate Config is also used by Wing to wrap around beautifully with Axolotl
- Estimate Memory goes through those calculations we went through earlier
- Accelerate Launch is how we run our script :) 

Launching distributed training sucks
There are different ways you can do it.... and all have slightly different commands
![[Pasted image 20240521114302.png]]
python script.py
torchrun and deepspeed are the two main commands you can use to run
- torchrun run on a single computer with 2 gpus, my script

This is a lot of different commands to remember
So `accelerate launch` is here to handle this for you

![[Pasted image 20240521114339.png]]

![[Pasted image 20240521114439.png]]
Accelerate: We want a low level way that is device agnostic (mac, windows) and compute agnostic (gpu, cpu, tpu). It does so in a minimally intrusive and ideally not-very-complex manner
- You create an accelerator, and have it prepare all your things (right code)

![[Pasted image 20240521114524.png]]

![[Pasted image 20240521114646.png]]
We try hard to make sure that 
Instead we map the forward pass with autocast to just convert the gradients; this leads to stable training and better finetuning later on
- If you go to bf16, you're stuck in bf16. There was a whole big issue where the quality of finetuning models wasn't goign well in `trasnforemrs`, and this was the cause.

![[Pasted image 20240521114726.png]]
You might have heard from something called TransformerEngine or MSAMP, where we ... do training in 8 BIT (raw native 8 bit)
- A lot of mistakes we see people do with this is that they do the prior thing of converting the model into bf16 and and then train
Heard that even this can go bad

![[Pasted image 20240521114942.png]]

![[Pasted image 20240521114949.png]]

---
Sam question:
DeepSpeed vs FSDP vs accelerate

When you want to do multi-gpu training in axolotl, you have to supply a config file...

----
# Training on Modal

We were pretty selective about the tools we brought in to the conference; we're only talking about tools we like

![[Pasted image 20240521115736.png]]

To experience the magic of modal... What Hamel likes to show people first is the /docs/guide/webhooks page, where you can change the code and see it change in production in real time. It's an iterative, interesting thing.


Modal - Axolotl Fine Tuning
- ![[Pasted image 20240521115952.png]]
- This tutorial sort of wraps Axolotl
- When you run the training, it automatically merges the LoRA back into the base model by default (you can turn this off)
- you have to pass a --data flag instead of relying on the config
- The Deepspeed config comes from the ___ repo instead
- Sort of like the beginner's way to use Axolotl; it's something to try first -- you can tweak it, change the code...



Recap: Language Models allow us to estimate the probability of a piece of text by factorizing that probability into the product of conditional probabilities preceding tokens, giving *their* preceding tokens.

For our [[N-Gram]] models, we used the Markov Assumption, where the next state depends only on the previous state (where we extended previous state to include one token (Unigram), 2 tokens (Bi-Gram), ... all the way up to N-grams).

![[Pasted image 20240521210529.png]]
Above:
- We count up the number of occurrences of "students opened their $w_j$"
- The denominator is crucial because it normalizes the count, ensuring that the probability is conditioned on the prefix. It effectively answers the question: out of all the times the prefix "students opened their" occurs, how many times does it continue with the specific word $w_j$

If we were using a *Bigram* model, we would ignore "students opened" and just look at P(w_j|their)

There's a reason few problems with using N-Gram models:
1. Sparsity problem; Let's say that "Students opened their laptop" never occurred in the training set? Then it'll be 0 in our numerator from the above picture. But what if it's a reasonable sentence? This can be somewhat ameliorated using [[Label Smoothing]]. There are problems with this too -- if you don't observe instances of "Students opened their laptop," you really have no idea whether the word is being used correctly or not; there's only so much you can do if you know nothing about a word or the context in which it occurs. This is also true of word types that just don't occur in the vocabulary. A "solution" there is to use an UNK token.
2. They're not great at modeling long-distance dependencies in language if the dependency is long than N tokens away (in an N-gram).
3. The storage cost of an N-gram model! For a bi-gram model, we have these big tables where the rows are the prefix, and the columns are the next word in the sentence. Every time we want to add more context (eg by going from a bigram to a trigram), that table size increases exponentially!
4. We treat all words/prefixes independently of eachother! "Students opened their {blank}" and "pupils opened their {blank}" have nothing to do with eachother, even though they're very similar, semantically!

On the plus side, N-gram models are very interpretable.

For *neural* language models, to combat issues like the ones above, we almost necessarily *have* to trade off a measure of interpretability.

![[Pasted image 20240521213240.png]]
N-gram models rely on the ==Bag of Words Assumption==, meaning that every single word type in the vocabulary and every single prefix are treated as one-hot vectors, where the length of our vector is the size of the vocabulary. Only one element o the vector is 1, and the rest are 0. If you think about an n-gram model, this makes sense; *Movies* and *Film* are completely independent of eachother, so maybe they should just be dissimilar in this way! But wait... ==all of these on-hot vectors are equally (dis)similar to eachother,== even when words like *Movies* and *Film* are actually kind of similar to eachother!

==What we want is a representation space in which words, phrases, sentences, etc. that are semantically similar also have similar representations to eachother!==









---


Python
Pytorch
Accelerate (A low-level wrapper; doesnt' assume anything the user wants to do; gives user complete freedom)
Other libraries built on top of Accelerate; Axolotl on the top, having the "highest level of abstraction", Transformers somewhere in-between. Everyone else just wraps around Accelerate and uses it with minimal headaches.