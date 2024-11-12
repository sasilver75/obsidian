Link: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms

--------

WE've recently seen the release of the latest [[LLaMA 3.2]] models, which include open-weight versions for the 1B and 3B LLMs, and two *multimodal models*!

Let's:
- Learn how multimodal LLMs function
- Review and summarize ~12 other recent multimodal papers and models published in recent weeks, to compare their approaches.

![[Pasted image 20241109013653.png|300]]
Above: A multimodal LLM that can accept different input modalities, and return text as the output modality.

## Use cases of multimodal LLMs
- A classic and intuitive application of multimodal LLMs is ==image captioning==: We provide an input image, and the model generates a description of the image.
- Extracting information from a PDF table and converting it into LaTeX or Markdown.

## Common approaches to building multimodal LLMs
There are two main approaches to building multimodal LLMs (terms he made up):
- Method A: ==Unified embedding decoder architecture== approach
- Method B: ==Cross-modality attention architecture== approach

![[Pasted image 20241109014012.png|600]]
Above:
- (Left) In the ==Unified Embedding Decoder Architecture==
	- We use a single decoder model; ==images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both== text and image input tokens together.
- (Right) In the ==Cross-Modality Attention Architecture==
	- Employs a cross-attention mechanism to ==integrate image and text embeddings directly within the attention layer==.

### Method A: Unified Embedding Decoder Architecture

![[Pasted image 20241109014611.png|400]]
Above: The ==unified embedding decoder architecture==, which uses an unmodified decoder-style LLM ([[GPT-2]], [[Phi-3]], [[Gemma]], [[LLaMA 3.2]]) that receives inputs consisting of both image tokens *and* text tokens!
- Image is converted into embedding vectors.
- For the text input, it's usually tokenized (e.g. using [[Byte-Pair Encoding]]) and then passed through an embedding layer.

![[Pasted image 20241109120842.png|300]]
Above: The standard process for tokenizing text and converting it into token embedding vectors, which are subsequently passed to an LLM during training and inference.

Similarly, image embeddings are generated using an image encoder model (instead of a tokenizer), as shown in the figure below:

![[Pasted image 20241109121002.png|300]]
Above: Illustration of the process for encoding an image into image patch embeddings.

What happens inside the image encoder?
![[Pasted image 20241109121028.png|300]]
We divide it into smaller patches (reminiscent of tokenization); these patches are then encoded by a pretrained vision transformer (ViT).
- ==NOTE==: ViTs are often used for classification tasks, which is why he has that classification head in the figure above; but for a VLM, we just need the image encoder part.

What's the role of this ==linear projection module== in the previous figure?
- It's a ==single  linear layer== -- the purpose is to project the image patches (which get flattened into a vector) into an embedding size compatible with the transformer encoder.
	- So an image patch might be flattened into a 256-dimensional vector, and then up-projected to a 768-dimensional vector.

![[Pasted image 20241109122153.png]]
Above: A linear projection layer projected flattened image patches from a 256-dimensional space to a 768-dimensional embedding space.

Let's compare the two encoders:
![[Pasted image 20241109122404.png|300]]
As you can see above, we have an additional ==*projector* module== added to the end of the end of the  image encoder. 
- This projector is usually just another *linear projection* layer that's similar to the one explained earlier! 
- The purpose is to ==project the image encoder inputs into a dimension that matches the dimensions of the embedded text tokens==.

Now that we have image patch embeddings that have the same embedding dimension as the text token embeddings, we can simply concatenate them as text input to the LLM:
![[Pasted image 20241109122746.png|200]]
See that we just prepend the text-embedding-dimensioned image patch embeddings to the sequence of text token embeddings, and then process them as usual in (e.g.) a decoder-only language model.
- A Popular choice for these image encoders are models like [[CLIP]] or [[OpenCLIP]]

==NOTE:== There are versions of Method A that operate *directly on the patches*, such as [[Fuyu]]
![[Pasted image 20241109123120.png]]
Above: See that [[Adept]]'s [[Fuyu]] VLM operates directly on image patches without an image encoder.
- Instead, they pass the input patches directly into a linear projection (embedding layer) to learn its own image patch embeddings, rather than relying on an additional pretrained image encoder like other models do.


### Method B: Cross-Modality Attention Architecture
- Let's talk about an alternative way (to the ==Unified Embedding Decoder Architecture==) approach -- we'll call this the ==Cross-Modality Attention Architecture==.

![[Pasted image 20241109123907.png|300]]
Above: See in this architecture that we use the ==same image encoding setup that we discussed previously,== but instead of encoding the patches as *input to the LLM,* we instead connect input patches in the ==multi-head attention layer== via a [[Cross-Attention]] mechanism.
- This idea goes all the way back to the original transformer architecture in the 2017 Attention is All You Need paper, which was developed for language translation. This consists of a text encoder that takes the sentence to be translated and generates the translation via the text decoder.


## Unified decoder and cross-attention model training
- Now that we've talked about the two major design choices, let's briefly talk about how we deal with the three major components during model training.

![[Pasted image 20241109124323.png|500]]
Above: 
- See that the ==projector in the image encoder== (that changes the dimensionality of the image encoder output tokens into some that's compatible with the downstream method) is ==initialized with random weights==, and trained.
- See that the ==image encoder== is usually a ==pretrained== vision transformer (eg [[CLIP]]) that ==remains frozen== during the entire training.
- See that the ==language model== itself is initialized with an instruction-tuned LLM that is ==frozen during pretraining, and later unfrozen during multimodal instruction-finetuning==.

In ==Stage 1== (multimodal pretraining), the image encoder and LLM parts are frozen, focusing only on the projector (a linear layer of small multi-layer perceptron).
In ==Stage 2== (multimodal instruction finetuning), the LLM is often unfrozen to allow for more comprehensive updates.
- ==NOTE==: For the cross-attention based models (Method B, above), the cross-attention layers are *unfrozen* throughout the *entire training process!*

After introducing these two approaches, (Method A and Method B), you might wonder which is more effective!
- ==Method A==: Typically *easier to implement* since it doesn't require modifications to the LLM architecture itself.
- ==Method B==: Often considered ***more computationally efficient*** because it doesn't overload the input context with additional image tokens, and introduces them later in the cross-attention layers instead.
	- Also ***maintains the text-only performance of the original LLM*** if the LLM parameters are kept frozen during training.


## Recent Multimodal Models and Methods

### The LLaMA 3 Herd of Models
- The [[LLaMA 3.2]] models were originally announced and made available on September 25.

![[Pasted image 20241109125616.png|500]]
Above: (The video and speech parts are visually occluded to focus attention on the image part)
- See that it uses a classic 2020 image transformer variant.
- See that this is an example of Method B, where we're cross-attending to the image representation in the language model.
	- ==NOTE:== The Meta researchers took the ==opposite of the usual approach== (of freezing the image encoder and only updating the LLM) by *updating the image encoder, but freezing the language model!*
	- They write that this is intentional, and done to ==preserve the text-only capabilities== of the LLM, so that the L3.2 11B and L3.2 90B can be used as drop-in replacements for the L3.1 8B and L3.1 70B text-only models on text tasks.
- Instead of adopting a pretrained image encoder like [[CLIP]], the researchers ==used a [[Vision Transformer|ViT]] that they *trained from scratch==!* They adopted the ViT-H/14 variant of the classic vision transformer, then pretrained the ViT on a dataset of 2.5B image-text pairs over five epochs. This is done before connecting the image encoder to the LLM.
- ==They only add cross-attention layers every fourth transformer block, since they add a substantial amount of parameters==.
	- For the 8B model, this adds 3B parameters; for the 70B model, this adds 20B parameters
		- (Aren't the image encoder parms included? I guess that's only 630M params dedicated to that)

### Molmo and PixMo: Open weights and open data for SoTA Multimodal Models
- In the [[Molmo]] paper (September 25, 2024), it promises to open-source not only the model weights but also dataset and code, similar to their work with the language-only [[OLMo]] LLM. 
	- This is great for LLM research as it allows us to do things like run ablation studies and reproduce results on the same dataset.
![[Pasted image 20241109131735.png|400]]
Above: 
- It looks to me that Molmo follows Method A, where the image and text embeddings are processed together, at the "bottom" of the model.
- The Molmo team streamlined the training process by avoiding multiple pretraining stages, instead choosing a simple pipeline that ==updated all LLM/Connector/Image encoder==.

  
The Molmo team offers several options for the base LLM:
- OLMo-7B-1024 (a fully open model backbone),
- OLMoE-1B-7B (a mixture-of-experts architecture; the most efficient model),
- Qwen2 7B (an open-weight model that performs better than OLMo-7B-1024),
- Qwen2 72B (an open-weight model and the best-performing model)


### NVLM: Open Frontier-Class Multimodal LLMs
- In [[NVLM]] (September 17, 2024), instead of focusing on a single approach, we explore both methods:
	- Method A: The Unified Encoding Decoder Architecture
	- Method B: The Cross-Modality Attention Architecture
- Additionally, they develop a hybrid approach (==NVLM-H==), and provide a comparison of all three methods.
![[Pasted image 20241109132524.png|500]]
Above:
- See that they use a (larger) 6B-param [[Vision Transformer|ViT]], InternViT (the encoder from [[InternVL 1.5]])
- It seems like their "downsampling" and MLP operations are the connector/projector/adapter.
- See that they then use those adapter image encoder tokens in 3 methods:
	- Method A (Unified Encoding Decoder Architecture)
	- Method B (Cross-Modality Attention Architecture)
	- "Method C" (their hybrid approach)

- Similar to [[Molmo]], they begin with text-only LLM rather than pretraining a multimodal model from scratch.
	- Interesting that he makes a comment that Molmo might have used Qwen2-72B base

### Qwen2-VL
- In [[Qwen 2 VL]] (October 3, 2024), the core of the work is around their so-called "==Naive Dynamic Resolution==" mechanism, which lets the model to handle images of varying resolutions without simple downsampling, enabling the input of images in their original resolution.
	- ((This seems like a huge deal!))
![[Pasted image 20241109135347.png|400]]
The native resolution input is implemented via a modified ViT by removing original absolute positional embeddings and introducing 2D-[[Rotary Positional Embedding|RoPE]].

Training consists of three stages:
1. Pretraining only the image encoder
2. Unfreezing all parameters (including the LLM)
3. Freezing the image encoder and instruction-finetuning only the LLM


### Pixtral 12B
- In [[Pixtral]] 12B (September 17, 2024), the first multimodal model from [[Mistral]].
- There's no technical paper or report available, but the team shared some interesting bits from the blogpost:
	- Trained an image encoder from scratch with 400M parameters instead of using a pretrained one.
	- Used the 12B param [[Mistral NeMo]] model as the language model.
- ==Interestingly, Pixtral also supports variable image sizes natively, as illustrated in the figure below!==

![[Pasted image 20241109140137.png|400]]

### MM1.5 Paper

### Aria: An open multimodal native MoE

### Baichuan-Omni

### Emu3: NTP is all you need

### Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
- With the [[Janus]] paper from [[DeepSeek]] (October 17, 2024), a frameowrk that unifies multimodal ***understanding*** and ***generation*** tasks within a single LLM backbone...
- Key feature:
	- Decoupling of visual encoding pathways to address the distinct requirements of *understanding* and *generation* tasks!
	- Researchers: "==Image understanding requires high-dimensional semantic representations, while generation tasks require local information and global consistency in images.=="
	- Researchers separate these pathways
- Model employs [[SigLIP]] vision encoder, and for image generation it utilizes a Vector-Quanitzed (VQ) tokenizer to handle the generation process.
![[Pasted image 20241109140524.png|400]]
Above: So is Janus an image generation model? Oh, it can do both, natively? That's pretty crazy. Without a tool use call out to other models? Whoah.

![[Pasted image 20241109140608.png|400]]

# Recap

![[Pasted image 20241109140629.png]]









