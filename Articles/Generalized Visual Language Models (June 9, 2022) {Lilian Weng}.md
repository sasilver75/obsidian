https://lilianweng.github.io/posts/2022-06-09-vlm/
Note that this is an article from June 2022, so it's not going to be "modern." It's basically going to be about models like [[CLIP]], [[Flamingo]], [[Contrastive Captioner|CoCa]], etc.

---

In this post, we'll talk about *one approach* for solving visual=language tasks, which is to ==extend pre-trained generalized LMs to be capable of consuming visual signals==. ((This is going to be in=contrast to later works like Chameleon that do vision/language training from the start.))

Lilian groups [[VLM|Vision-Language Model]]s (VLMs) into four buckets:
1. Translating images into embedding features that can be ==jointly trained== with token embeddings.
2. Learning good image embeddings that can work as a ==prefix== for a frozen, pre-trained LM.
3. Using a specially-designed ==cross-attention== mechanism to fuse visual information into layers of the language model.
4. Combine vision and language models without and training.

## (1/4) Jointly Training with Image and Text



## (2/4) Learned Image Embedding as (Frozen) LM Prefix



## (3/4) Text-Image Cross-Attention Fuse Mechanisms


## (4/4) No Training
