---
aliases:
  - ViT
---
October 22, 2020
[[Google Research]]
Paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
#zotero 
Takeaway: Applying a pure transformer approach directly to sequences of image patches to perform image classification. Split an image into patches and provide the sequence of linear embeddings of these patches as input to a Transformer (which we train with a supervised image-classification objective), where they're treated the same way as tokens (words) in an NLP application.

Examples: [[Swin]], [[DeiT]], EfficientViT, ...

---

Notes:
- Authors note that when trained on mid-sized datasets like ImageNet, ViTs yield modest accuracies, still a few % below ResNets of comparable size (this is expected, given CNNs' inductive biases, which Transformers lack) -- but when we train on larger datasets (14M-300M images), we find that Transformers beat the biases of models like ResNets.
- ==Process==: We reshape three-dimensional (channels) images into a sequence of ***two-dimensional*** patches, which are then flattened and piped through a trainable linear projection to map it into the $D$-dimensional space that the Transformer (**encoder-only**) uses as its latent vector size throughout its layers. We also add positional embeddings to patch embeddings to retain positional information. Similar to BERT's use of a `[CLASS]` token, we prepend a learnable embedding to the sequence of embedded patches, whose state at the output of the Transformer encoder serves as the image representation. We then attach a classification head on top of this, implemented via an MLP.
	- The positional encodings are purely learned during training, rather than using fixed functions like the frequency-based ones used in the original Transformer.
- The Transformer encoder consists of the layers of multi-headed self-attention and MLP blocks, with [[Layer Normalization|LayerNorm]] applied before every block, and residual connections after each block. The ML has two layers with a [[GeLU]] nonlinearity
- Authors train a ViT-Base (86M), ViT-Large (307M) and ViT-Huge (632M) models on  ~300M images from ImageNet and JFT.


Before reading Questions: 
- It seems interesting that we're able to use 16x16 image patches as  tokens. That seems to imply a vocabulary size of all possible permutations of 16x16x3=768 positions, permuted, which is enormous?
	- The answer seems to be that we embed the patches via a linear transformation; so it's more like we treat them as token embeddings, rather than as discrete tokens, if that makes sense. They call these patch embeddings. There's no discrete "vocabulary" like there is in language.

# Paper Figures
![[Pasted image 20240506133340.png]]

![[Pasted image 20240506140831.png]]

![[Pasted image 20240506140925.png|100]]




# Non-Paper Figures 
![[Pasted image 20240123104746.png]]
Above: Many cutting-edge CV models consist of multiple stages:
- Backbone extracts the features
- Neck refines the features
- Head makes the detection for the task

