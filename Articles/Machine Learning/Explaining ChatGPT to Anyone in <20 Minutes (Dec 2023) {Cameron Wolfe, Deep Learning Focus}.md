#article 
#premium 

The goal of this paper is to be a primer that brings everyone up to speed on how LLMs work.

# The core components of Generative LLMs

Top-level view:
1. Transformer architecture
2. Language model pretraining
3. The alignment process

## Transformer Architecture
- Encoder-Decoder Architecture
	- The ==Encoder== looks at the full sequence of text provided as input and builds a representation of this text and outputs this representation.
	- The ==Decoder== ingests the encoder's representation of the English sentence, generating a Chinese translation.
- Decoder-only Architecture
	- Generative LLMs typically use a decoder-only architecture, which eliminates the encoder from the transformer, leaving only the decoder. Given that the decoder's role in the transformer is to generate textual output, we can intuitively understand why generative LLMs only use the decoder component, as opposed to only using the encoder component (which produces a non-human-interpretable sequence of embeddings).
- Constructing the input
	- First, we have to tokenize our raw text, breaking it into a sequence of tokens (i.e. words or subwords). After tokenizing the input sequence, we convert each token into an associated, unique vector representation, forming a list of vectors, each corresponding to each token.
	- These token vectors are lists of number that quantitatively describe each token.
	- In the original Transformer, we upgrade each token representation with positional information by adding separate positional embeddings/vectors to them.
	- We learn these initial token vectors' vector representation (the numbers at each position in the vectors) through trial and error, after being randomly initialized. 
		- In the original Transformer, the positional vectors that we add are *not* learned, but there are versions of position vector that *do* involve learning.
- Processing the input
	- Each "block" of the decoder-only transformer takes a list of token vectors as input, producing a list of list of transformed token vectors (of the same size) as output. This is composed of two operations:
		- A *Masked/Causal Self-Attention* operation, where we transform each vector by considering tokens that precede it in the sequence.
		- *Feed-forward transformation*: Transforms each token vector individually via a sequence of linear layers and non-linearities.
	- Together, masked self-attention and feed-forward transformations allow us to create a rich representation of any textual sequence.

## Language Model Pretraining

> Self-supervised learning obtains supervisory signals from the data itself, often leveraging the underlying structure in the data. The general technique of self-supervised learning is to predict any unobserved or hidden parts/properties of the input from any observed or hidden part of the input.

Self-supervised pretraining observes to the idea of using signals that are already present in raw data to train a machine learning model. In the case of generative language models, the most commonly-used objective for self-supervised learning is next token prediction, known as the ==standard language-modeling objective==.
- This objective is the core of all generative language models.

LLM Pretraining
- We first curate a large corpus of raw text to use as a dataset.
- Starting from a randomly-initialized model, we then pretrain the LLM by iteratively performing the following steps:
	1. Sample a sequence of raw text from the dataset
	2. Pass this textual sequence through the decoder-only transformer
	3. Train the model to accurately predict the next token at each position within the sequence.
- The underlying objective is self-supervised here -- because the "label" that we train the model to predict (the next token) is *always, inherently present* within the underlying data. As a result, the model can learn from massive amounts of data without the need for human annotation.

Inference process:
- The model follows an autoregressive process, comprised of the following steps:
	1. Take an initial textual sequence as input
	2. Predict the next token
	3. Add this token to the input sequence
	4. Repeat steps 2-3 until a terminal/stop token (\[EOS\]) is predicted.

Keys to success:
- Larger models are better; increasing the size of parameters yields smooth increase in performance.
- More data; Increasing the size of the underlying model alone is suboptimal.

A decoder-only transformer that has been pretrained over a large textual corpus is typically referred to as a base model. Notable examples include [[GPT-3]], [[Chinchilla]], and [[LLaMA 2]].


## The Alignment Process
- What we really want is for our models to be correct, helpful, honest, and harmless. But the pre-training process has a misalignment -- it doesn't train models to be that -- it instead just trains the model to be good at predicting the next token *in our training dataset.* 
- The ==Alignment Process== is a process of fine-tuning (adjusting) model parameters to guide the model in such a direction that it scores highly in human preference evaluations, namely, embodying the ability to:
	1. Follow detailed instructions
	2. Object constraints in the prompt
	3. Avoid harmful or dishonest outputs
	4. Avoid hallucination
	5. Anything else we might want!
- The objectives of the alignment process are typically referred to as the ==alignment criteria==, which we must define at the outset of the alignment process. These might include things like *helpfulness* or *harmlessness*.

![[Pasted image 20240427170542.png]]

- ==[[Supervised Fine-Tuning|SFT]]== is simple to understand:
	- Based on the same objectives as pretraining (next token prediction), but on a golden dataset of human-agent interactions (either real ones or synthetically-generated ones) that exemplify the characteristics that we want to emulate
- Manually creating such a high-quality dataset for supervised fine-tuning can be difficult and expensive; we use use techniques like [[Reinforcement Learning from Human Feedback|RLHF]] to bootstrap an evaluation process:
	- Starting with a set of prompts and the LLM(s), we generate two+ responses to each prompt. 
	- A human annotator(s) rank responses to each prompt based on the defined alignment criteria.
	- We use this labeled data to train a ==reward model== that accurately predicts human preference scores given a prompt and a model's response to that prompt.
		- This is key to *scaling* RLHF; *actual* human evaluation is expensive and slow, so we attempt to create a classifier that closely correlates with actual human preference annotators -- then we use *that* to train our model.
	- Then, once we have a reward model, we can use a reinforcement learning algorithm (eg PPO) during the second phase of RLHF to finetune the LLM to maximize human preference scores, as predicted by the reward model.

This approach was critical to the success of [[InstructGPT]], which was the predecessor to [[ChatGPT]]
![[Pasted image 20240427171051.png]]

What's next for explanations?
- In-Context Learning: Writing a prompt to solve the desired task
- Finetuning: Performing further training on data for the desired task
- Retrieval-augmentation
- Decoding strategies (eg CoT, BeamSearch, Speculative Decoding)














