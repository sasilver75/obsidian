Link: https://www.determined.ai/blog/multimodal-llms

----

## Flamingo
- [[Flamingo]] is a multimodal LLM presented in 2022. Here's how the vision and language components work:
	- The vision encoder converts images or videos into embeddings. 
	- Theses embeddings can vary in size depending on the dimensions of input images (or length of input videos), so another component called the ==Perceiver Resampler== converts these embeddings to a common fixed length.
	- The LM takes in both text and the fixed-length vision embeddings from the Perceived Resampler. The vision embeddings are used in multiple cross-attention blocks, which learn to weigh the importance of different parts of the vision embedding, given the current text.
![[Pasted image 20240612223557.png]]
Training occurs in three steps:
- Vision encoder is pretrained using [[CLIP]] (CLIP actually trains both a vision encoder and a text encoder, so the text encoder from this step is discarded).
- The LM is a [[Chinchilla]] model.
- In the third stage, untrained cross-attention blocks are *inserted into the language model*, and an untrained Perceiver Resampler is inserted between the vision encoder and the language model.
	- We freeze the weights of the vision encoder and language model, and update just the Perceiver Resampler and cross-attention blocks.

After training, Flamingo is able to performa a variety of vision tasks, including answering questions about images in a conversational format!
![[Pasted image 20240612223823.png]]


## What is CLIP?
- As mentioned in the previous section, Flamingo uses [[CLIP]] in its pretraining stage.
- CLIP isn't a multimodal VLM; instead, ==it's a *training methodology* that produces separate vision and text models== with powerful downstream capabilities -- it stands of Contrastive Language-Image Pre-Training
	- The model architecture consisting of an image encoder and text encoder. 
	- The two encoders are trained on batches of image-text pairs, in which the text describes the image. The encoders are trained such that:
		- For each image-text pair, the image and text embeddings are "close" to eachother.
		- For all non-matching image-text pairs, the image and text embeddings are far from eachother.

- Note that there's many ways to measure the distance between two embeddings. Some common ways are euclidean distance and cosine similarity; CLIP uses the latter.

![[Pasted image 20240612225438.png]]

At a high level, CLIP learns joint image-text embedding space, which basically means that the similarity between images and text can be directly computed. It turns out that training models with this goal makes them generally very useful, including in the context of multimodal LLMs.


## BLIP-2
- [[BLIP-2]] is a multimodal LLM released in early 2023.
- Like Flamingo, it contains pretrained LLM and image encoder components.
	- *Unlike* Flamingo, both the image encoder *and* LLM are left untouched (after pretraining). (Recall that there were cross-attention blocks inserted into Flamingo's LLM)
- To connect the image encoder to the LLM, BLIP-2 uses a =="Q-Former"==, which consists of two components:
	- The visual component receives a set of learnable embeddings, and the output of the frozen image encoder. As in Flamingo, the image embeddings are fed into cross-attention layers.
	- The text component receives text.
![[Pasted image 20240612230251.png]]
Training occurs in two stages:
- In stage 1, the two components of the Q-Former are trained on three objectives, which actually originate from the [[BLIP]] paper:
	- Image-text contrastive learning (similar to CLIP, but with some differences)
	- Image-grounded text generation (generating captions of images)
	- Image-text matching (a binary classification task where for each image-text pair, the model has to answer 1 to indicate a match and 0 otherwise)
- In stage 2, the full model is constructed by inserting a projection layer between the Q-Former and the LLM. This projection layer transformers the Q-Formers embeddings to have lengths that are LLM-compatible.
	- The full model is then tasked with describing images. During this stage, the image encoder and LLM remain frozen, and only the Q-former and projection layer are trained.
![[Pasted image 20240612230834.png]]
The authors show that BLIP-2 can outperform Flamingo on various visual question-answering tasks, but with only a fraction of the trainable parameters, making it easier and more cost-effective to train.


## LLaVA
- [[LLaVA]] is a multimodal LLM that was released in 2023; the architecture is quite simple:
	- The vision encoder is pretrained using CLIP
	- The LLM is a pretrained [[Vicuna]] model
	- The vision encoder is connected to the LLM by a single projection layer.
- Notice the simplicity of the component and their interaction, in comparison with BLIP-2 and Flamingo!
- There are two training stages:
	- In stage 1, the training objective is image captioning. The vision encoder and LLM are frozen, so only the projection layer is trained.
	- In stage 2, the LLM and projection layer are finetuned on a partially-synthetic instruction-following dataset. It's partially synthetic because  it's generated with the help of GPT-4 (??)
![[Pasted image 20240612231141.png]]
LLaVA illustrates that even a simple architecture can achieve excellent results when trained on partially synthetic data.















