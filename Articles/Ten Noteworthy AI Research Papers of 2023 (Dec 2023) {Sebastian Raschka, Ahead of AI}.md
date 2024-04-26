#article 
Link: https://magazine.sebastianraschka.com/p/10-ai-research-papers-2023

--------

This is Sebastian's list of "best" papers in 2023, a year for which he can't recall a time when the field was more popular and rapidly evolving.


# (1) Pythia - Insights from Large-Scale Training Runes
- [[Pythia]] (Apr 2023)
- The researchers originally released ==a range 8 LLMs== ranging from 70M to 12B parameters (==both the weights and data were publicly released==, along with training details, analyses, and insights)
- Some questions addressed by the paper:
	1. Does pretraining on duplicated data (eg >1 epoch) make a difference?
		- Duplication doesn't benefit or hurt performance
	2. Does training order influence memorization?
		- It does not
	3. Does pretrained term frequency influence task performance?
		- Yes, few-short accuracy tends to be higher for terms that occur more frequently
	4. Does increasing the batch size affect training efficiency and model convergence?
		- Doubling batch size halves training time and doesn't hurt convergence.

The small LLMs in the <1B range are nice templates for small studies and tinkering, or starters for pretraining experiments!


# (2) LLaMA 2: Open Foundation and Fine-Tuned Chat Models
- [[LLaMA 2]] (Large Language Model Meta AI) (Jul 2023)
- The LLaMA 2 models range from B to 70B parameters, and are still ==among the most capable and widely-used openly-available models==. The license also permits use in commercial applications!
- The model is differentiated by coming not just as a standard pretrained model, but also ==providing chat models that have been finetuned via [[Reinforcement Learning from Human Feedback|RLHF]]== and instruction-tuning [[Supervised Fine-Tuning|SFT]].
![[Pasted image 20240409112139.png|450]]
In the in-depth 77-page research report, the authors also nicely illustrate the evolution of the Llama 2 70B chat models, tracing their journey from the initial supervised finetuning to the final RLHF finetuning stage with [[Proximal Policy Optimization|PPO]]
![[Pasted image 20240409112223.png|450]]
The above chart shows the improvement in both helpfulness and harmless of the model over iterated finetuning (using both SFT and RLHF), as judged by both Meta reward models and Meta-external reward models (GPT-4)

# (3) QLoRA: Efficient Finetuning of a Quantized LLM
- [[Quantized Low-Rank Adaptation|QLoRA]] (May 2023)
- One of the favorite techniques of the LLM research community because it made the popular [[Low-Rank Adaptation|LoRA]] technique more memory efficient! ==It means that you can fit larger models onto smaller GPUs, in short.==
![[Pasted image 20240409112525.png]]
Above:
- We decompose the weight update matrix into two lower-rank matrices that can be multiplied together to "recover" the weight update matrix.

The standard LoRA modifies a pretrained LM by ==adding low-rank matrices to the weights of the model's layer==; these matrices are smaller and therefore require fewer resources to update during finetuning.

==In QLoRA, these low-rank matrices are quantized==, meaning their numerical precision is reduced. This is done by mapping the continuous range of values in these matrices to a limited set of discrete levels, reducing the model's memory footprint and computational demands.

According to the paper, it reduces the memory requirements of a 65B Llama model to fit onto a single 48 GB GPU (like an A100)!

==QLoRA is a handy tool for reducing GPU memory requirements during finetuning==


# (4) Bloomberg GPT: An LLM for Finance
- March 2023
- Sebastian is including this because, while it didn't result in a groundbreaking new insight, ==it was an interesting case study where someone pretrained a relatively large LLM on a *domain specific dataset*==
- It was a 50B parameter model for finance, trained on 363 billion tokens from finance data and 345 billion tokens frmo a general, publicly available dataset.
- Note: the smaller AdaptLLM-7B model, which cost only $100, ended up outperforming BloombergGPT on one dataset, and nearly matching its performance on three other finance datasets.


# (5) Direct Preference Optimization (DPO): Your Language Model is secretly a Reward Model
- [[Direct Preference Optimization]] (May 2023)
- Quick Recap of Reward Models
	- We usually finetune our chatbots in the following manner:
		1. Supervised finetuning over a dataset containing instructions and desired responses
		2. Reward modeling, where human raters provide feedback on the model's outputs. This data is used to create a *reward model*, which learns to predict what kinds of outputs are to be preferred.
		3. [[Proximal Policy Optimization]] (PPO): The model generates outputs, the reward model scores each output, and we use the PPO Reinforcement Learning algorithm to use these scores to adjust the model's policy (ie its weights/parameters).
- Above: [[Reinforcement Learning from Human Feedback|RLHF]] is popular and effective, but it's expensive (human preference annotation) and fickle (PPO) to implement.
- The DPO paper ==introduces an algorithm that optimizes language models to align with human preferences *without* explicit reward modeling or reinforcement learning==. Instead, DPO ==uses a simple classification objective==.
	- We still keep the supervised finetuning step (step 1 above), but we replace steps 2 and 3 with a *single step* to further finetune the model on preference data.

==The appeal of DPO lies in the simplicity of the method== -- the scarcity of chat models trained with RLHF can likely be attributed to the complexity of the RLHF aproach.


# (6) Mistral 7B
- [[Mistral]] (Oct 2023)
- Introduces a compact yet powerful language model that, ==despite its relatively modest size of 7 billion tokens, outperforms its larger counterparts, such as the 13B Llama 2 model, in various benchmarks==.
- Why it's so good is unclear, but it might likely be due to its training data, which isn't disclosed.
- Architecture wise:
	- The model shares [[Grouped Query Attention]] with Llama 2
	- Uses [[Sliding Window Attention]] (2019) to save memory and improve computational throughput for faster training.
		- Essentially a fixed-sized attention block that allows a current token to attend only a specific number of previous tokens (instead of all previous tokens).
			- ((So you're lobotomizing your long-range dependency capabilities to make training faster? Okay))
- One reason that Mistral 7B is an influential model is that it served as the base model for [[Zephyr]] B, as mentioned in the DPO section, this was the first model trained with DPO to outperform other alternatives!
- Another noteworthy model derived from Mistral 7B is the [[Mixtral]] 8x7B model: Mistral [[Mixture-of-Experts]] (MoE), which matches or exceeds the performance of the even larger Llama-2-70B model!

# (7) Orca 2: Teaching Small Language Models how to Reason
- [[Orca 2]] (Nov 2023)
- Combines several concepts and ideas
	- ==Distilling data from large, capable models like GPT-4 to create a synthetic dataset to train small but capable LLMs==
		- This idea was described in the [[Self-Instruct]] paper
		- Earlier in 2023, [[Alpaca]] (a [[LLaMA]] model finetuned on ChatGPT outputs) really popularized this approach.

This works in a four-step process:
1. Seed task pool with a set of human-written instructions (175 in this case) and sample instructions
2. Use a pretrained LLM (eg GPT-3) to determine the task category
3. Given the new instruction, let a pretrained LLM generate the response
4. Collect, prune, and filter the responses before adding them to the task pool

![[Pasted image 20240409123016.png]]
Above: I believe this is from the Self-Instruct paper (the reason why they separate it into determining whether it's classification or not is that they want to make sure to generate the appropriate balance of classes, in the case of classifications, which doesn't always happen if you just greedily sample from a language model)

The other idea might not be surprising but is worth highlighting: high-quality data is important for finetuning.
- The [[LIMA]] paper proposed a human-generated high-quality dataset consisting of only 1k training examples that can be used to finetuning to output the same model finetuned on 50k ChatGPT-generated responses.
	- ((TLDR: High-quality data, in this case generated by humans, is incredibly important for finetuning))

Unlike previous research, Orca 2 aims to teach "small" (7B, 13B) LLMs various reasoning techniques (step-by-step reasoning, recall-then-generate, etc.) and to help them determine the most effective strategy for each task.

This led to Orca 2 outperforming similar-sized models noticeably and even achieve results comparable to models 5-10 times larger.

While we haven't seen any extensive studies on this, the Orca 2 approach might also be able to address the issue of synthetic data highlighted in the earlier paper "==The False Promise of Imitating Proprietary LLMs==" (which was targeted at the large number of models finetuning LLaMA with ChatGPT-generated outputs)
- Here, the authors investigate the finetuning of weaker language models to imitate stronger proprietary models like ChatGPT, using examples like Alpaca and Self-Instruct
- Initially, there were promising results (re: human evaluation of responses vs ChatGPT), but more follow-up evaluations revealed ==these imitation models only *SEEMED* to perform well to the observer, but often generated factually-incorrect responses!==

# (8) ConvNets match Vision Transformers at Scale
- October 2023
- In recent years, Sebastian (author) has only worked with large language transformers or vision transformers (ViTs) due to their good performance.
- ==What's particularly appealing about transformers for CV is that pretrained ViTs are even easier to finetune than CNNs!==
- The "ConvNets match vision transformers at scale" paper showed that convolutional NNs (CNNs) are, in fact, competitive with ViTs when given access to large enough datasets.
	- ((This points to the idea that architectures aren't *really* the most important thing to improve/innovate on. It's all about FLOPs, then DATA. See: https://twitter.com/tszzl/status/1736286837822595177))


# (9) Segment Anything
- [[Segment Anything Model]] (SAM, Meta, April 2023)
- Object recognition, segmentation in images and videos, classification and generative modeling, are main research fields in CV.
- To briefly highlight the difference between these two tasks: 
	- Object detection is about *predicting bounding boxes* and the associated labels
	- Segmentation *classifies each pixel* to distinguish between foreground and background objects

Meta's Segment Anything model is a notable milestone for open-source and image-segmentation research.
- Researchers used licensed and privacy-respecting images, so the model can be open-sourced without major copywrite concerns.

SAM consists of three main components
1. An ==image encoder== utilizing a Masked [[Autoencoder]] based on a pretrained [[Vision Transformer]] (ViT) that can handle high-resolution inputs. The encoder is run once per image, and can be applied before prompting the model.
2. A ==prompt encoder== that handles two types of prompts:
	1. Sparse (points, boxes, text)
		- Point and boxes are represented by positional encodings combined with learned embeddings for each prompt type
		- Free-form text uses an off-the-shelf text encoder from CLIP
	2. Dense (masks)
		- Dense prompts (i.e. masks) are embedded using convolutions and summed element-wise with the image embedding.
3. A ==mask decoder== maps the image embedding, prompt embeddings, and an output token to a mask.
	- This is a decoder-style transformer that compute the mask foreground probability at each image location.

# (10) Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models 
- (April 2023)
- Eh. It's video generation with Diffusion








