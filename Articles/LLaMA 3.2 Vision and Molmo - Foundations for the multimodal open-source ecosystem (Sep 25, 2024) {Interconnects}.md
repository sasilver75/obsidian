Link: https://www.interconnects.ai/p/molmo-and-llama-3-vision

----

Frontier labs and small albs are trying to even ==define== what it means for multimodals to see the world.

Most multimodal research is through ==late fusion models,== where the model is initialized from a language backbone, and an image encoder. 
- This has been popular with [[Molmo]] and [[LLaMA 3.2]] V

The promise of data scaling through ==early fusion models== hasn't yet emerged ((I think [[Chameleon]] is one of these?))

Basic questions like:
- How do standard text benchmarks like [[GSM8K]] and [[IFEval]] degrade with multimodal training?

Standard evaluations that largely assess the model's knowledge, like [[MMLU]] are known to be ~unchanged in the best visual finetunes.

[[Allen Institute|AI2]] released [[Molmo]]:
- Molmo 72B, built on [[Qwen 2]] 72B
- Molmo-7B-D, built on Qwen 2 7B
- Molmo-O, built on forthcoming [[OLMo]] 7B version
- Molmo-E, built on [[OLMoE]]

> Our model architecture follows the simple and standard design of combining a language model with an image encoder. It consists of four components: (1) a ==pre-processor that converts the input image into a set of multiscale, multi-crop images==; (2) a ==ViT image encoder that independently maps each of these images into a set of vision tokens==; (3) a ==connector that projects the vision tokens to the language model's input dimension with an MLP and then pools the vision tokens to reduce their count==; and (4) a ==decoder-only Transformer LLM==.

![[Pasted image 20241008181758.png|350]]

![[Pasted image 20241008182056.png]]

My read, which is to be expected based on the organizations’ goals, is that Llama 3.2 V is a better text model, maybe even much better, but Molmo is a better image model. This is particularly true with the features like pointing and reading clocks that Molmo does so well at.

==Molmo has a capability that none of its peer models have — the ability to point at pixels in a referenced image. For example, I asked [Molmo where the bike](https://molmo.allenai.org/share/963de0d0-9069-4a14-ad5a-8e4bc0863136) was in a photo of myself.==



![[Pasted image 20241008184121.png]]
Interesting technique here that he uses to invoke the "vision model" version of Sonnet; he attaches an image and says "ignore the image" (which of course the model can't do; the image will affect the activations at the point of fusion, and a different image might give the right answer, who knows. I'm sure you could adversarially find a picture that gives you the right answer.

![[Pasted image 20241008184628.png|400]]

The only application that isn’t solved yet which is certain to be important to the future multimodal language models is web element understanding. Web agents are one of the last integration hurdles limiting a mass rollout of generative AI products.













