#article 
Link: https://cameronrwolfe.substack.com/p/t5-text-to-text-transformers-part

---------
![[Pasted image 20240304221500.png]]

The transfer learning paradigm is comprised of two main stages:
1. We pretrain a deep neural network over a bunch of data
2. We fine-tune this model (ie. train it some more) over a specific downstream dataset.

The exact implementation of these stages may take many forms!
- In computer vision, for example, we often pre-train models on the ImageNet dataset using a supervised learning objective, and then do supervised fine-tuning on some downstream dataset that we're actually trying to solve.
- In natural language processing, we often perform *self-supervised pre-training* over an unlabeled corpus of text, before similarly fine-tuning (there's some more nuance here than in CV).

Combining large, DNNs with massive (pre-)training datasets often leads to impressive results.
- Given the raw amount of textual data freely available on the internet, we can simply download a massive textual corpus, pre-train a large neural net on this data, and then fine-tune the model on a variety of downstream tasks.

This type of large-scale transfer learning approach was initially explored by BERT, which pretrained a transformer encoder over unlabeled data using a masking objective, and then fine-tuned on downstream language tasks.
- ((I think Jeremy Howard would have aneurism for not mentioning the previous ULMFit paper))

The success of BERT can't be overstated, and so the NLP community began to heavily investigate the topic of transfer learning, leading to the proposal of many new extensions and improvements.

====The Text-to-Text Transformers Model ([[T5]]) proposed a unified framework for studying transfer learning approaches in NLP, letting us analyze different settings and derive a set of best practices==! This set of best practices comprises T5, a state-of-the-art model and training framework for language tasks.

# Relevant History and Context
- T5 reformulates existing transfer learning techniques into a unified format, compares them, and determines best practices to arrive at a high-performing result.

But first, what is transfer learning and different variants of the transformer architecture.
## What is transfer learning?
- If we want to train a neural network to solve some task, we have two basic options:
	1. Training from scratch for the target task
	2. Transfer learning (pre-training on a separate dataset, and then finetuning on the target task)

Typically, pre-training is performed over a dataset that's much larger than the downstream, target dataset.
- In computer vision, for example, we might pre-train a model over ImageNet, and then fine-tune on a smaller dataset like CIFAR-10/100.
- For NLP, the story is a bit different -- typically we use *self-supervised* pre-training objectives with unlabeled text.


## Different Transformer Architectures
- The transformer, as originally proposed, uses an encoder-decoder architecture, as shown below:
![[Pasted image 20240304223102.png|150]]
- The transformer uses an [[Encoder-Decoder Architecture]], as shown above. 
- However, the full encoder-decoder transformer architecture isn't our only option!
	1. [[Bidirectional Encoder Representations from Transformers|BERT]] uses an [[Encoder-Only Architecture]]
	2. Most modern language models use [[Decoder-Only Architecture]] transformers.

Let's look quickly at these two variants:

### A primer on Self-Attention
- The self-attention operation take a sequence of tokens as input and produces a new sequence of *transformed* token vectors with the same length as an output.
- Each entry of this new sequence is an attention-weighted average of value vectors derived from the input sequence.

Single Stack or Double stack?
- The original transformer architecture uses two "stacks" of transformer layers; see above.
- The first stack (the *==encoder== module*) is comprised of several blocks that contain ==bidirectional== self-attention and a feed-forward neural network.
- The second stack (*the ==decoder== module*) is comprised of several blocks, but it uses ==masked== self-attention. and also has an added ==cross attention== mechanism that considers activations within the attention encoder layer while performing self attention.


The transformer was originally used for sequence-to-sequence tasks. For other tasks, single stack transformer models have become popular:
- Language models use a decoder-only architecture
- BERT-style models use an encoder-only architecture

#### Attention Masks
- Variants of the transformer architecture have one major distinction: the *type of masking used in their attention layers!*
- Here, when we refer to ==masking==, we are referring to certain tokens being masked (or ignored) during the computation of self-attention.
- Put simply, certain tokens may look only at a select portion of other tokens in the full input sequence.

Encoder-only models leverage bidirectional (or fully-visible) self-attention, which considers all tokens within the entire sequence during self-attention. Here, each token representation in self-attention is computed as a weighted average of all other tokens in the sequence.

In contrast, Decoder-only models use causal self-attention, where each token only considers tokens that come before it in the sequence.

We also can adopt a hybrid approach by defining a "prefix." More specifically ,we can perform bidirectional self-attention for a group of tokens at the beginning of the sequence (i.e. a prefix), then perform causal self-attention for the rest of the ...

But certain applications (eg language modeling) *require* that we use causal self-attention during training to prevent the transformer from "looking in the future" (i.e. just copying the correct token when generating output).


What does T5 use?
- Although the analysis considers many transformer architectures, ==the primary model used for T5 is a standard encoder-decoder architecture==!
- Aside from a few small modifications, this model is quite similar to the transformer as it was originally proposed.
- Encoder-only architecture are not explored because they are designed for token or sequence level classification and not generative tasks like translation or summarization.

# BERT: Transfer Learning for NLP
- Transfer learning in NLP typically uses recurrent neural networks pre-trained with a causal language modeling objective.
	- But everything changed with the proposal of [[Bidirectional Encoder Representations from Transformers|BERT]], a transformer model that's pre-trained with a self-supervised objective (Masked Language Modeling/Cloze or Next Sentence Prediction), and then can be fine-tuned for use in a downstream task that you care about.
	- At the time of its proposal in late 2018, BERT set SoTA in nearly all NLP tasks that were considered (11).

To make this a bit more specific, BERT relies on a "denoising" objective, called [[Masked Language Model]]ing, during pre-training.
This might sound a bit complicated, but the idea is simple -- we just:
1. Mask some tokens in the input sequence by replacing them with a special MASK token
2. Process the corrupted/modified sequence with BERT
3. Train BERT to accurately predict the masked tokens

The exact implementation is a bit more complicated. We select 15% of tokens at random, then either replace them with MASK tokens (90% probability) or a random token (10% probability).

### How is T5 related to BERT?
- The proposal of BERT showed that transfer learning is a useful approach for solving NLP problems!
- Many people began trying new techniques and performing improvements -- as a result, the field was overwhelmed with different options for performing transfer learning with BERT-like models. 
- T5 continues in this line of research, but ==T5 tries to analyze all of these different proposals using a unified framework, giving us a better picture of best practices for transfer learning in NLP. It then goes to train a final T5 model using all of the identified best practices, and reaches SoTA performance!== 

How is T5 related to LLMs?
- ==LLMs are great, but T5 exists in a relatively distinct area of tools and research; namely, it focuses mostly on models that explicitly process input with an encoder *before* generating output with some separate decoder.==


# T5: The Unified Text-to-Text Transformer 🎉
- The contribution of T5 isn't just a novel architecture or training methodology -- it's actually based entirely on existing techniques -- T5 considers all aspects of the transfer learning pipeline in NLP, such as:
	- Different (unlabeled) datasets
	- Pre-training objectives
	- Benchmarks
	- Fine-tuning methods
- The goal of T5 is to:
	1. Analyze these transfer learning setups
	2. Determine the most effective approaches

### Text-to-Text Framework
- ==T5 converts all text processing problems into a "text-to-text" format (taking text as input and producing text as output).==
	- This generic structure allows us to solve a variety of different tasks with a shared approach.
	- We just adopting a *prompting* approach and simply *ask* our language model to generate the answer in a textual format!

![[Pasted image 20240304234855.png|450]]

To make this a bit more concrete, all tasks being solved by T4 can be converted into a text-to-text format by:
1. Add a ==task-specific prefix== to the original input sequence
2. Feed this sequence to the transformer
3. Formulate the model's target as a textual sequence

Using this, we can easily perform tasks like summarization or translation.
It gets a little bit more complicated for problems like regression (i.e., we have to round real-valued outputs to the nearest decimal and treat it as a classification problem), but it tends to work well for a majority of linguistic tasks.

T5 is fine-tuned on each task that it solves.

# How is T5 studied?
- All analysis performed uses the unified, text-to-text framework described above, as it allows a variety of different language tasks to be converted into a shared format.

![[Pasted image 20240304235410.png]]

The Model
- As discussed previously, the transformer architecture as originally proposed contains both an encoder and a decoder module.
- Recent work on language modeling has explored architectural variants that are encoder or decoder-only.
	- [[Bidirectional Encoder Representations from Transformers|BERT]] is encoder-only
	- Most LLMs are decoder-only

- ==T5 uses an [[Encoder-Decoder Architecture]] that closely resembles the original transformer. The differences are:==
	1. [[Layer Normalization|LayerNorm]] is applied immediately before each attention and feed-forward transformation (i.e. outside the residual path)
	2. No additive bias is used for LayerNorm (We only use scale and eliminate the additive bias)
	3. A simple position embedding scheme is used that adds a scalar to the corresponding logit used to compute attention weights
	4. Dropout is applied throughout the network (attention weights, feed forward network, skip connections, etc.)

Pre-training dataset
- T5 is pretrained over the [[C4|Colossal Clean Crawled Corpus]] (C4) corpus of "relatively clean" English text. Authors construct their pre-training dataset from this, using a limited set of filtering rules.
- Notably, C4 was later used as a subset of the [[MassiveText]] dataset used to pre-train [[Gopher]] and [[Chinchilla]].

Experimental Setup
- T5 is pretrained over C4 and then fine-tuned to solve a variety of downstream-tasks -- however, the exact settings used within this framework are variable.

# Takeaways
- This post has covered all preliminary information related to the T5 model, including important background information and the basic experimental framework that is used.
- In the next post, we will cover details of the extensive analysis performed in T5, which uncovers best practices for transfer learning in NLP.

For now, the major takeaways are:
1. Transfer learning is powerful
2. What comes after BERT? (T5 attempts to unify all the follow-up work and analysis after the release of BERT)
3. Generic task formulation (A generic text-to-text framework that can restructure any language task into textual input and output. This is done by appending a task-specific prefix to the textual input and using the decoder module of T5 to generate text corresponding to the desired target)















