https://www.artfintel.com/p/papers-ive-read-this-week-vision

I this after reading [[Understanding Multimodal LLMs (Nov 3, 2024) {Sebastian Raschka}]] article, at the recommendation of Xeo in Interconnects discord.

----

"This post was inspired by Claude being so ridiculously good at converting equation screenshots to LaTeX, which made mew ant to understand how LLMs can be so good at understanding pictures and doing fancy OCR -- is reading text from pictures a solved problem? How do VLMs work?"
- Spoiler: It's actually super straight forward -- it turns out that using some vision encoder (eg a ViT) to convert images into features, patchifying it, and concatenating the resulting sequence with the text embeddings, is basically enough.

## The evolution of multimodal model architectures
- He starts his learning with a [survey paper from Purdue](https://arxiv.org/abs/2405.17927), where he groups multimodal architectures into four categories:
	- Type A and Type B, which combine multimodal inputs within the ==internal layers of the model==
		- Type A employs ==cross-attention==
		- Type B uses ==custom layers for modality fusion==
	- Type C and Type D, which ==combine the modalities at the input stage==
		- Type C uses ==modality-specific encoders==
		- Type D uses tokenizers to ==convert every mode into tokens, and then processes them together==

Examples of models which fall into the various categories:
1. Type A (Internally uses cross attention to combine): [[Flamingo]]
2. Type B (Internally uses custom layers for modality fusion): [[CogVLM]], [[MoE-LLaVA]]
3. Type C (At input stage, uses modality-specific encoders): [[DeepSeek-VL]], [[LLaVA]], [[Sphinx]], [[Emu]], [[Qwen VL]]
4. Type D (At input stage, converts all into tokens; processes together): [[LaVIT]], [[Unified-IO]]

==Contemporary open-source models are almost *all* doing type C!==
- Type D is somewhat common in video models (eg MagViT2), but most multimodal papers aren't bothering to convert image features into tokens, but passing the patchified features directly into the decoder.
	- ((Unclear what he means by this... Is a patchified embedding not a token? What is a "vision" token that he'd expect?))

In deciding between ==late/deep fusion== (where modalities are combined within the internal layers of the model) and ==early fusion== (where they're combined at the input), the larger labs will likely focus on the most general approach (eg [[Bitter Lesson]]), which says that we should *learn* as much of our structure as possible, rather than imposing pre-determined inductive biases.

Survey paper is a useful overview, but it does get bogged down in coming up with a detailed taxonomy of VLMs, which seems to be of questionable utility... but it's a good intro.

## Flamingo
- [[Flamingo]], paper [here](https://arxiv.org/abs/2204.14198), an April 2022 paper from [[DeepMind]], and one of the early multimodal LMs.
- It focused on enabling zero-shot adaptation to novel tasks that use ==text and image inputs==! Reached SOTA in 6 tasks.
- Architecture combines a pretrained+frozen vision encoder with a similar frozen+pretrained language model, and then they ==only== train dense cross-attention layers, along with a ==[Perceiver](https://arxiv.org/abs/2103.03206)== resampling layer on top of the vision encoder.
	- This is ==a much more complicated architecture than we'll in the later models==, which all tend to be decoders with minor tweaks.

![[Pasted image 20241112144134.png|600]]
Above: Flamingo architecture
- Core idea: A perceiver resampler on top of a frozen vision encoder. Out of this is used to condition a frozen LM by freshly-initialized gated cross-attention layers.

The Flamingo model seemed to work well, but was very complex -- with the benefit of hindsight, it's interesting how unnecessary the complexity was, as none of the subsequent SOTA models used a similar architecture.

## QWEN-VL
- [[Qwen VL]], August 2023 ([Paper](https://arxiv.org/abs/2308.12966)) from [[Alibaba Research]]... This is when we start to see architectures/training methodologies arise that are very similar to the state of the art in Q3 2024.
- The Qwen-VL series of models is based on the [[Qwen]] LLM, and adds visual abilities with ViT-bigG, which is initialized from [[OpenCLIP]] model. 
	- They resize images to 224x224, split it into patches, ((embed it)), and then use a vision-language adapter to compress the image features ((assumedly into the same-dimension space as the text tokens?)). 
	- They then use a single layer of Cross Attention, which uses a number of learned embeddings as query vectors, and the image features from the visual encoder as keys for cross-attention, outputting a sequence of length 256.
	- They use 2D absolute positional encodings on the output of the cross-attention layer.
	- They have three stages of training during which thy freeze various parts of the model.
		- Stage 1: Pretraining (LM frozen, Cross Attention and ViT unfrozen)
		- Stage 2: Multitask training (All unfrozen)
		- Stage 3: Supervised finetuning (LM and Cross Attention unfrozen, ViT frozen)

![[Pasted image 20241112145001.png|600]]
They add special tokens (\<img\>, \<\img\>) to the sequence to denote the start/end of image content, and also train the model with bounding boxes, including them ==as text tokens==, which are the in teh standard way, but with two types of special tokens:
- \<box\> and \<\box\> to denote the coordinates
- \<ref\> and \<\ref\> to denote text description corresponding to a given bounding box.
((==It's interesting to me that  these bounding box and "refs" for bounding box captions are included as part of the text tokens, rather than image tokens. It makes sense, I suppose!==))

The model is pretrained on web-scraped image-text pairs, and then trained on high-quality, fine-grained annotation data. They have additional SFT stage that they use to create a chat model. The result was strong and achieved SOTA in many tasks.


## CogVLM
- Published in late 2023, [[CogVLM]] used a ==frozen, trained language model and image encoder, and combines the two with a trainable "visual expert module" in the attention and FFNN layers==, enabling vision features without sacrificing NLP performance.
- Likely implemented contemporaneously with Qwen-VL, but definitely has more in common with the *pre-[[Qwen]]* architectures, like [[Flamingo]].

![[Pasted image 20241112145719.png|600]]
- Above: In [[CogVLM]], ==they add two trainable layers to each transformer block== :
	- An MLP
	- A QKV Matrix
- Both are initialized from their pretrained counter parts int he language model ((but used to process the image information, somehow? I imagine they must be feeding the same set of tokens into the language and image 'branches" though?))
- Interestingly, they assign all the image tokens a *==single positional ID for RoPE==*... with the logic that visual tokens already encapsulate positional information when inputted into the ViT... and that by adding additional position information, the query would focus more on the closer tokens, i.e. the lower part of an image.

Authors did a lot of ablation experiments.


## DeepSeek-VL
- [[DeepSeek-VL]] was released in March 2024, and is the [[DeepSeek]] take on a VLM. It's a refinement of the [[Qwen VL]] approach.
- For the visual feature module, the authors use a ==*hybrid encoder*==, which has ==BOTH== a text-aligned encoder for coarse semantic extraction at 384x384 resolution, with a high--resolution encoder that operates at 1024x1024 resolution!
	- ((Are these the sizes to which refinement happens? Are the patch sizes the same between each?))
	- The high-resolution encoder is based on [[Segment Anything Model|SAM]]-B ("Base" size) , and [[SigLIP]] is used for the low-resolution image inputs.
		- SigLIP can be viewed as a "better CLIP," so this is a modernization/improvement on what the Qwen-VL did.
- ==Authors use a two-layer MLP as a vision-language adapter, with two distinct single-layer MLPs processing the high- and low- resolution features separately.== These features are then ==CONCATENATED== together, and transformed into the LLM's input space through another layer of MLP.

1. Authors keep the LLM and vision encoders frozen while they train the adapter module
2. Then they jointly train the LLM + VL adaptor on interleaved VL+text-only sequences
	- ((This is making me think -- it really is important that the data is interleaved, and that we don't just consider that the image tokens should be prepended to the text tokens. If my user interaction is asking a series of questions about different models, the position of my images relative to text is obviously important!))
3. Then they finetune the entire model on chat data, unfreezing everything.

![[Pasted image 20241112164612.png|600]]
Above: The [[DeepSeekVL]] training process

This model was SOTA against other open-source 7B models, but was unsurprisingly not as good against proprietary LMMS like GPT-4V or Gemeni Pro.
- Their model also didn't see significant degradation on language model benchmarks, ==which is a problem that plagues VLMs, which tend to have rapidly degraded performance on LLM tasks!==
	- This model is a good counterexample for the frozen text models we've seen consistently.


## Chameleon
- Published by [[Meta AI Research]] in May 2024, [[Chameleon]] is an example of a "modern" multimodal model which uses [[Early Fusion]] to treat *all modalities as discrete tokens* ((that seems normal to me...)). 
- The loss here is the standard autoregressive loss:
	- `<start_image>` and `<end_image` tokens used to insert the image tokens into the sequence input.
	- Achieves SOTA on a number of benchmark and is competitive with text-only models of similar sizes on text tasks (Mixtral 8x7B, Gemeni-Pro)
![[Pasted image 20241112165519.png|500]]
- The authors train the model on a variety of orderings, including text-only, text-image pairs, and fully-interleaved text-image documents.

For the image encoder, they use the [[Make-A-Scene]] model, which encodes a 512x512 image into a 1024 discrete tokens, using a codebook with size 8192.
- ((It's interesting that they choose to use a model that has a codebook (basically a vocabulary for image patch embeddings)))
- The authors note that this tokenizer is particularly bad at reconstructing images with a large amount of text, which is to be expected given the limited codebook size.


## PaliGemma
- Is July 2024's [[PaliGemma]] model from [[DeepMind|Google DeepMind]], we continue to demonstrate the superiority of the [[SigLIP]] model, and we combine a 400M SigLIP model with a 2b Gemma model into ~3b 






















