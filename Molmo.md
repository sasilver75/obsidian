September 25, 2024
[[Allen Institute|AI2]]
[Molmo Blog Post](https://molmo.allenai.org/blog)

References:
- [Interconnects](https://www.interconnects.ai/p/molmo-and-llama-3-vision) blog post

Multimodal model from Allen Institute, with ==open data== and ==open code==.
- Molmo's training data is called [[PixMo]], also described in the blog post. 
	- "The longevity of the PixMo datasets "All of these models use vision encoders without open data (mostly CLIP from OpenAI), but open-source alternatives exist. Releasing image data is far riskier (for things no one wants to deal with, like CSAM) and complex."

Model Family
- MolmoE-1B, a mixture of experts model with 1B (active) 7B (total)
- Molmo-7B-O, our most open 7B model
- Molmo-7B-D, our demo model
- Molmo-72B, our best model

Blog Posts
> Our model architecture follows the simple and standard design of combining a language model with an image encoder. It consists of four components: (1) a ==pre-processor== that converts the input image into a set of multiscale, multi-crop images; (2) a ViT ==image encoder== that independently maps each of these images into a set of vision tokens; (3) a ==connector== that projects the vision tokens to the language model's input dimension with an MLP and then pools the vision tokens to reduce their count; and (4) a ==decoder-only Transformer== LLM.
> For the vision encoder, all of our released models use OpenAI's ViT-L/14 336px CLIP model which provides consistently good results.
> For the LLM, we have trained models on a variety of choices at different scales and degrees of openness including: the fully open-weight and data OLMo-7B-1024 (using the October, 2024 pre-released weights, which will be public at a later date), the efficient fully open-weight and data OLMoE-1B-7B-0924, open-weight Qwen2 7B, open-weight Qwen2 72B, open-weight Mistral 7B, open-weight Gemma2 9B, and Phi 3 Medium). Today we are releasing 4 samples from this family.

Key Results
> - Our most efficient Molmo model MolmoE-1B, based on our fully open OLMoE-1B-7B mixture-of-experts LLM, nearly matches the performance of GPT-4V on both academic benchmarks and human evaluation.
> - Our two Molmo-7B models perform comfortably between GPT-4V and GPT-4o on both academic benchmarks and human evaluation, and significantly outperform recently released Pixtral 12B models on both benchmarks.
> - Our best-in-class Molmo model, Molmo-72B, achieves the highest academic benchmark score and ranks second on human evaluation, ==just slightly behind GPT-4o==.
> - Our ==best Molmo model also outperforms== several state-of-the-art proprietary systems, including ==Gemini 1.5 Pro and Flash and Claude 3.5 Sonnet==.


-----
Notes 
![[Pasted image 20241008182129.png|300]]
> Molmo has a capability that none of its peer models have — the ability to point at pixels in a referenced image. For example, I asked [Molmo where the bike](https://molmo.allenai.org/share/963de0d0-9069-4a14-ad5a-8e4bc0863136) was in a photo of myself.

> This sort of description is directly downstream of the new dataset, PixMo, used to train the Molmo models. PixMo be relevant far longer than these initial models. The innovation for the dataset is to have annotators respond to images in audio rather than text (similar to this [localized narratives paper](https://arxiv.org/abs/1912.03098)), which enabled them to be far more creative and descriptive in their annotations. In fact, the annotators enjoyed the tasks given to them so much (such as the pointing data) that they were actively asking for more tasks to do. _Unlocking_ the annotators to be extremely engaged is a goal for any human data pipeline, but not one I have seen, ever. This dataset has millions of examples across a wide variety of images.



# Non-Paper Figures

![[Pasted image 20240925115935.png]]
![[Pasted image 20240925115953.png]]![[Pasted image 20240925121931.png]]
