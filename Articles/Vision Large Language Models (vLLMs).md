Cameron R. Wolfe
https://cameronrwolfe.substack.com/p/vision-llms?utm_source=post-email-title&publication_id=1092659&post_id=158954054

March 31, 2025

---------------------------

vLLMs are LLMs that can ingest images and video as input in addition to text.
Recent OpenAI models support visual inputs, and Meta has released a vision-based variant of Llama 3, called Llama-3.2 Vision.

Let's go over how vLLMs work from first principles, starting with basic concepts and eventually seeing how Llama 3.2-Vision is practically implemented.


Recall:
- Transformer blocks are repeated and consist primarily of:
	- Self-Attention: Transformers each token vector based on the other tokens that are present in the sequence.
	- Feed-Forward Neural Networks: Transforms each token vector individually.

Most LLMs use a [[Decoder-Only Architecture]], but sometimes additional modules are added to the architecture to handle vision-based inputs.

[[Self-Attention]] computes attention over the tokens in a single sequence.
![[Pasted image 20250402162919.png|750]]


In contrast, [[Cross-Attention]] considers two sequences of tokens -- the tokens from the encoder and the tokens from the decoder -- and computes attention between these two sequences.
![[Pasted image 20250402162934.png|750]]
Key difference: Instead of computing all three of these matrices by linearly projecting a single sequence of token vectors, we linearly project TWO sequences of token bvectors!
- The query matrix is produced by linearly projecting the first sequence.
- Both key and values matrices are produced by linearly projecting the *second* sequence.
We're computing inter-sequence attention scores, forming a fused representation of the two input sequences!

![[Pasted image 20250402163814.png|700]]
Above: Integrating an image encoder features into an LLM using cross-attention.

Cross-Attention is used constantly in multimodal research.

____________

## Vision Transformers (ViT)
- The most commonly used architecture today for VLMs.
- We take a sequence of vectors as input and apply a series of transformer blocks containing the usual
	- Bidirectional self-attention.
	- Feed-forward transformation.

![[Pasted image 20250402164025.png|600]]

Handling input images:
- The input for a vision transformer is an image.
- To pass this image to our transformer, we need to convert the image into a list of vectors (just like we do for a sequence of text).
- For ViTs, we do this by segmenting an image into a set of ==patches== and ==flattening== each patch into a vector.
- Still, our vectors may not be the same size as expected by the transformer, so we may ==linearly project== them into the correct dimension.

![[Pasted image 20250402164201.png|800]]

- We add ==positional embeddings== to the vector for each patch. These capture the 2D position of each patch within an image.
- The output of this transformer architecture is a sequence of vectors for each patch that is of the same size as the input.
	- If we wanted to solve tasks like image classification, we can just add an additional classification module (e.g. a linear layer) to the end of hte model!

Why an encoder-only architecture?
- The reason for this is that ViTs aren't generative!
- For LLMs, we train models by the language modeling objective of next token prediction, using masked attention.
- ViTs, in contrast, should be able to look at the entire sequenceo f image patches to form a good representation of the image! 
	- We don't care about predicting the next patch in this inpunt sequence.

Training ViTs
- The original ViT model shares the same architecture as BERT, basically.
- ==All ViT  models are trained using supervised image classification on datasets of varying sizes==
- When ViTs are trained over small or mid-sized datasets (e.g. ImageNet), they ==perform comparably or slightly worse-than== ResNets! 
- But ==ViTs begin to shine when pretrained over much larger datasets==, outperforming ResNets.


# Contrastive Language-Image Pre-Training (CLIP)
- The standard ViT is trained over a large dataset of supervised image classification examples.
- These models perform best when trained over a massive volume of usually ==human-annotated data==, which is ==difficult and expensive to maintain==.
- In contrast, in [[CLIP]], authors explore an alternative using image-alt-text-caption pairs, which are readily available in large amounts online!

![[Pasted image 20250402165209.png]]

CLIP architecture:
- Image Encoder
- Text Encoder

Given an image-text pair as input, we pass each component separately to their corresponding encoder to get a dense vector representation.
- The ==image encoder== is a standard [[Vision Transformer|ViT]]
- The ==text encoder== is a [[Decoder-Only Architecture]] transformer (e.g. a GPT-style LLM).

[[Contrastive Loss|Contrastive Learning]]
- We could classify the images based on the words in the caption or use the LLM component of the architecture to generate captions based on the image.
- The key contribution is the idea of using a simple and efficient training objective -- based upon ideas from contrastive learning.
- 








 





























