September 25, 2024
[[Allen Institute|AI2]]
[Molmo Blog Post](https://molmo.allenai.org/blog)

Multimodal model from Allen Institute, with ==open data== and ==open code==.
- Molmo's training data is called [[PixMo]], also described in the blog post.

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


# Non-Paper Figures

![[Pasted image 20240925115935.png]]
![[Pasted image 20240925115953.png]]![[Pasted image 20240925121931.png]]
