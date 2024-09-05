https://www.interconnects.ai/p/llama-405b-open-frontier-model

---

Meta released their [[LLaMA 3.1]] suite of LLaMA models today, coming in the sizes:
- 8B
- 70B
- 405B

The models are ==best-in-class open-weight models==, with the 405B being directly in the ballpark of Anthropic's Claude 3 and OpenAI's GPT-4o.

The model's architecture is so simple that simplicity is a talking point! It's a feedforward, dense transformer with many, many parameters trained on 15.6T carefully curated tokens of data.
- License allows for synthetic data creation, but comes with heavy branding terms.

This model showcases Meta's focus on scaling their *systems*, rather than following the path of MoE and distillation that OpenAI/Anthropic/Google have done for their flashy small models ([[Gemeni Flash]], Haiku, [[GPT-4o Mini]]).

Considering the 405B model:
![[Pasted image 20240723171654.png]]
See that in the categories in which LLaMA 3.1 405B loses, it's very close behind alternative models.
Authors said that they really focused on reasoning, tool use, and instruction following.

Considering the smaller 8B and 70B 3.1 models
![[Pasted image 20240723172011.png]]
See that the 8B and 70B models (esp. 8B) basically wipe the field compared with every other similar open model, notably the much-like [[Gemma]] series of models.

Authors release quantized models, such as the 405B model in FP8, so it can easily be run on a single node of 8x80GB A100s or H100s for inference.

Data is king in the model: [[Scale AI]] is credited for being the partner in post-training, in addition to a substantial amount of synthetic data, which is very similar to the [[Nemotron-4]] recipe.

---

Zuckerberg wrote an [essay](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/) on why open-source AI is the right direction to pursue, with three arguments:
1. Open-source AI for developers
2. ... for Meta
3. ... for the world

He's using craft storytelling to try and oversell the role of Meta. Meta is commoditizing their compliments. There's a long history of technology companies doing this. It works for LLaMA 3 scale, but it's not clear if it would still be reasonable to do for a 10x larger LLaMA 4.
- LLaMA 3 likely costs on the order of $100M, which is cheap for a company like Meta...
- $1B (10x) and higher would start to shift the shareholders.

Meta tries to sell LLaMa as the central point of open-source aI, but while a foundation model is a large piece of the open-source ecosystem, the tooling to modify it and open resources for training are just as important, and Meta doesn't own them.

Even if Meta doesn't create lock-in for users, each marginal model it releases puts real pressure on competitors. *Separating competitive pressure from branding is the hardest part of this Meta strategy -- they don't know which one will matter more.*

The LLaMA 3.1 license is a modification of the fairly restrictive LLaMA 3 license, keeping most of the core terms around commercial usage restrictions, naming restrictions, etc.
- ==The primary news is that the LLaMA 3.1 license allows for training on outputs (for synthetic data).==, and changes for downstream naming.
- ==Users can now train *other models than LLaMA models* on the outputs of LLaMA models.==
- Users must still name their downstream models "LLaMA-(your-model-name)", which is a slight change from LLaMA-3-xyz"
	- ==Meta is doing its best to absorb all of the work in the open-source language modeling community into the LLaMa Brand.==
	- In retrospect, the original LLaMA 3 license didn't make sense, because they were trying to proliferate LLaMA branding with the name, but they kneecapped one of the primary methods for distribution in synthetic data.
- Any derivative artifact (models, datasets) must be distributed with the LLaMA 3.1 license.
- ==Companies with > 700M active users at the time of release cannot use the model.==

Most companies will try to comply, but individuals playing with training synthetic models online will take this as free reign to use LLaMA 3.1 outputs to train open models.

The license was likely weakened from 3 -> 3.1 due to community pushback, so we'll see what future licenses look like.




