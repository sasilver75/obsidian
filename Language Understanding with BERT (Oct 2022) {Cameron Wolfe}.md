#article 
Link: https://cameronrwolfe.substack.com/p/language-understanding-with-bert

((==SAM NOTE==))
- This is from 2022, so it's a pretty historical article. There are some things in here that were maybe correct at the time (or maybe still correct, but they call it something different), but aren't obviously correct any more.

----------
![[Pasted image 20240304173621.png|600]]

# What is BERT?
- [[Bidirectional Encoder Representations from Transformers]] (BERT) is a popular deep learning model used for *numerous different language understanding tasks!*
- BERT shares the same architecture as a transformer encoder, as is extensively pre-trained on raw, unlabeled textual data using a *self-supervised learning objective*, before then being *fine-tuned* to solve downstream tasks (eg question answering, sentence classification, named entity recognition, etc.)
	- BERT obtained a new SoTA on 11 different language understanding tasks, prompting a nearly-instant rise to fame that's lasted ever since!

The incredible effectiveness of BERT arises from:
1. ==Pre-training over large amounts of raw textual== data via self-supervised learning.
2. Crafting ==rich, bidirectional feature representation== of each within a sequence.

Previous work had demonstrated that language modeling tasks benefit from pre-training over large textual corpora, but BERT extending this idea by crafting a simple, yet effective suite of [[Self-Supervised Learning]] pretraining tasks that enable relevant features to be learned.

Additionally, BERT moved away from the common practice of using *unidirectional* self-attention, which was commonly adopted to enable language modeling-style pre-training within such language understanding tasks.
- Instead, BERT leveraged bidirectional self-attention within each of its layers, revealing that ==bidirectional pre-training is pivotal to achieving robust language representations==.

**BERT is very useful!**
- The simple answer is that BERT is incredibly generic -- this simple model architecture ==can be used to solve a surprising number of different tasks within SoTA accuracy==, including both token-level (e.g. named entity recognition) and sentence-level (eg sentiment classification) language understanding tasks.
- Additionally, its use has been expanded beyond the NLP domain to solve problems like multi-modal classification, semantic segmentation, and more! ==It is the Swiss Army Knife of deep learning!==

# Building Blocks of BERT
- Before overviewing the specifics of the BERT architecture, it's important to build an understanding of core components and ideas upon which BERT is built! These main concepts can be boiled down to the following:
	1. [[Bidirectional Attention]] (Self)
	2. Transformer Encoders
	3. Self-Supervised Learning

#### SElf-Attention
- At a high-level, self-attention is a non-linear transformation that takes a sequence of "tokens" as input, each of which is represented as a vector.
	- ((Sam: This intuitively about why it's non-linear? Non-linear means that it applies the same (?) transformation to each point in space... but if each point in space is a token embedding, we know that they all end up in meaningfully different points in space on the other side of attention block.))
- We represent this input sequence as a sequence of vectors (a matrix). Then, these token matrices are *transformed*, resulting in a *new* matrix of token representation.

What happens in this transformation?
- For each individual token vector, self-attention does the following:
	1. ==Compares that token to every other token in the sequence==
		- ((We compare the Query projection of the current token with the Key projection of the other tokens))
	2. ==Computes an *Attention score* for each of these pairs==
		- ((We do the `Softmax(QK/sqrt(d))` for each pair of tokens))
	3. ==Adapts the current token's representation based on the other tokens in the sequence, weighted by the attention score==!
		- ((The output vector for the current embedding vector is the Attention-Score-weighted sum of the Value vectors in the sequence, I believe?))

Intuitively, ==self-attention just adapts each token's vector representation based on the other tokens in the sequence, forming a more context-aware representation== -- see below!
![[Pasted image 20240304201542.png|350]]

#### Multiple Attention Heads
- Self-attention is usually implemented in a multi-headed a fashion, where multiple self-attention modules are applied *in parallel*, before having their outputs *concatenated!*
- ==The benefit of such a multi-headed approach lies in the fact that each head within a multi-headed attention layer can learn different attention patterns within the underlying sequence==, since the softmax in a single head pushes you towards attending to only a small number of tokens.

#### Unidirectional vs Bidirectional
- When crafting a context-aware representation of each token in the sequence there are two options in defining this context:
	1. Consider all tokens ([[Bidirectional Attention]])
		- Crafts each token representation based on all other tokens within a sequence.
	2. Consider all tokens to the left of the current tokens ([[Masked Attention]])
		- Ensures that the representation of each token only depends on those tokens that *precede it* in the sequence!
		- Such a modification for applications like language modeling that *shouldn't be allowed* to look "look forward" to predict the next word.

## Transformer Encoders
- The transformer architecture typically has two components - an encoder and a decoder.
- ==BERT, however, only uses the encoder component of the transformer.==
- As can be seen, the transformer encoder is just several repeated layers with (bidirectional, multi-headed) self-attention and feed-forward transformations, each followed by **layer normalization** and **residual connection**!
![[Pasted image 20240304205938.png|200]]

The two components of the Transformer generally have different purposes:
1. The ==encoder==: Leverages bidirectional self-attention to encode the raw input sequence into a sequence of discriminative token features.
2. The ==decoder==: Takes the rich, encoded representation and decodes it into a new, desired sequence (e.g. a translation of the original sequence).

# Self-Supervised Learning
- One of the key components to BERT's incredible performance is its ability to be pre-trained in a self-supervised manner.
- At a high level, such training is valuable because it can be performed over raw, unlabeled text -- because data of this kind is widely available online, a large corpus of textual data can relatively easily be gathered for pre-training, enabling BERT to learn from datasets magnitudes larger than the past.

Though many examples of self-supervised training objectives exist, some examples include:
1. [[Masked Language Model]]ing (MLM): Masking/removing certain words in a sentence and trying to predict them.
	- This is also called a *Cloze* objective
2. Next Sentence Prediction (NSP): Given a pair of sentences, predicting whether these sentences follow eachother in the text corpus or not.

==Neither of these tasks require any human annotation -- The labels are naturally present in the data!==

#### Q: Is this unsupervised learning?
- No! 
- Both unsupervised and self-supervised learning do not leverage labeled data, but while unsupervised learning is focused on discovering and leverage latent patterns in the data itself, self-supervised learning instead finds some *supervised training signal* that is already present in the data, and uses it for training.


# How BERT Actually Works

## BERT's Architecture
- As mentioned before, the architecture of BERT is just the encoder portion of a transformer model.
- The main distinction between BERT and previously-proposed language understanding models (eg OpenAI GPT) lies in the use of bidirectional self-attention, rather than unidirectional self-attention, which as a result lets it learn richer representations of the data.

> Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model *or* the shallow concatenation between *both* a left-to-right and a right-to-left model.

The process of getting data into a BERT-compatible sequence from raw text is as follows:
1. Tokenization
	- Raw textual data is broken into individual tokens or elements that represent words or parts of words.
2. Inserting "special" tokens
	- BERT's input sequence begins with a CLS token and ends with a SEP token. If two consecutive sentences are used, another SEP token is placed between them.
3. Embedding
	- Convert each vector into its corresponding (eg) WordPiece embedding vector.
4. Additive Embeddings
	- The input data is now a sequence of vectors. Learnable embeddings are added to each element in this sequence, representing the element's position in the sequence and whether it is part of the first or second sentence. 
	- Such information is needed because self-attention cannot otherwise distinguish an element's position within a sequence.

By following these steps, raw textual data is converted into a sequence of vectors that can be digested by BERT.


# Training BERT
- The training process for BERT proceeds in two steps:
	1. Pre-trained
	2. Fine-tuning

The architecture is nearly identical between these steps, though some small, task-specific modules may be used (e.g. MLM and NSP both use a single, additional classification head/layer on the end of the network)

#### Pretraining
- The BERT model is pretrained over unlabeled data using two different tasks:
	1. Masked Language Modeling (MLM) (Cloze task)
	2. Next sentence Prediction (NSP)
- Notably, ==BERT cannot be meaningfully trained with the typical language modeling objective of next-token prediction==, because the use of bidirectional self-attention (rather than masked self-attention) would just allow BERT to cheat by simply observing and copying this next token.

![[Pasted image 20240304213831.png]]
- Above: Next Sentence Prediction (NSP) pretraining task
	- The NSP task is simple -- consecutive sequences from pre-training corpus are passed into BERT, and 50% of the time the second sentence is replaced with another random sentence. then, the final representation of the CLS token, after being processed by BERT, is passed though a classification module that predicts whether the inputted sentences are an actual match.

![[Pasted image 20240304213945.png]]
Above: Masked Language Modeling (MLM) pretraining task
- Not a sequence-level task like NSP above. It randomly masks 15% of the tokens within the input sequence by replacing them with special MASK tokens.
- Then, the final representation for each of these MASK tokens is passed through a classification layer to predict the masked word.
	- Instead of always masking the tokens this way, though, authors replace each token with MASK 80% of the time, a random token 10% of the time, and the original token 10$ of the time.... this is to avoid issues with the MASK token being present in pre-training but not fine-tuning.


## Fine-tuning
- The self-attention mechanism within BERT is constructed such that modeling different kinds of downstream tasks is as simple as possible. In most cases, one just has to match the input-output structure of the task to the input-output structure of BERT, then perform finetuning over all parameters.
	- Examples:
		- Token-level tasks
		- Sentence/Document level tasks
		- Text pair tasks
- The general task structures listed above should demonstrate that BERT is a versatile model.
- During fine-tuning, all BERT parameters are trained in an end-to-end fashion.


## BERT might be the best thing since sliced bread?
- The results achieved with BERT on various different tasks are outlined below.
- You might notice something interesting about BERT's performance in these experiments -- it is never outperformed (except by humans) but only in certain cases.
- At the time of publication, BERT sets a new SoTA on eleven different NLP benchmarks.
- BERT is arelatively old model given the current pace of deep learning, but it's still simple and performant in a profound way.
	1. Bidirectional Self-Attention
	2. Self-Supervised Learning
- More recently, researchers have shown that the formulation of these self-supervised tasks themselves -- as opposed to just the massive amount of data used for pre-training -- are key to BERT's success.























