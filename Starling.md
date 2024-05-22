November, 2023
UC Berkeley
Blog: [Starling-7B: Increasing LLM Helpfulness and Harmlessness with RLAIF](https://starling.cs.berkeley.edu/)
#zotero 
Takeaway: The full paper isn't out yet, but this seems like a relatively normal finetuning of a 7B model ([[OpenChat]] 3.5 base -> Starling), except it was preference-tuned using the [[Nectar]] synthetic dataset, of 3.8M pairwise comparisons.

Zotero Note: There's no PDF (afaik), so the Zotero link links to an un-highlightable document. So I'll try to make my notes heavier here to be more comprehensive.

Additional note: The paper isn't completely done yet, they say -- they'll update it in the future, they say.

----

Notes: 
- Starling-7B is an open language model trained on [[Reinforcement Learning from from AI Feedback|RLAIF]]. The model harnesses the power of our new GPT-4-labeled ranking dataset, [[Nectar]], and our new reward training and policy tuning pipeline.
	- We release the reward model Starling-RM-7B-alpha as well as the language model Starling-LM-7B-alpha on HuggingFace.
- SFT datasets distilled from ChatGPT/GPT-4 have had remarkably effective ([[Alpaca]], [[Vicuna]], [[OpenHermes2.5 Dataset]], Openchat 3.5). 
- In contrast, the extent to which RLHF or RLAIF can enhance models when scaling high-quality *preference data* remains an open question. Earlier endeavors like [[Zephyr]]-7B, Neural-Chat-7B, [[Tulu 2]]-DPO-70B all employed [[Direct Preference Optimization|DPO]], but their performance in MT Bench/ChatBot Arena hasn't fully showcased RLHF's potential, compared to models like Open[[Hermes]] 2.5 and [[OpenChat]] 3.5.
- To facilitate more thorough research into RLHF, a high-quality ranking dataset specifically for chat is essential, so we release [[Nectar]], a GPT-4-labeled ranking dataset composed of 183K chat prompts. Each prompt includes 7 responses distilled from various models like GPT-4, GPT-3.5-instruct, GPT-3.5-turbo, Mistral-7B-Instruct, LLaMA2-7B, resulting in ==a total of 3.8M pairwise comparisons==. 
	- We also release our reward mode, Starling-RM-7B-alpha, trained with our K-wise loss on the Nectar dataset.
	- We then fine-tuned the [[OpenChat]] 3.5 LM using our learned reward model. Our team is actively exploring the various training methodologies for both the reward and language models, and will update.
- Authors complain about the HuggingFace [[OpenLLM Leaderboard]] as a benchmark for chat models, because, unlike [[AlpacaEval]] and [[MT-Bench]], it doesn't support custom chat templates.
- Authors note that (Goodhaet's Law) higher preference ranking by GPT-4 does not necessarily correlate with human preference -- specifically, that higher MT-Bench scores (as endorsed by GPT-4), don't automatically imply greater human favorability, especially when it comes to models with lower scores.
- Authors say that RLHF primarily enhances the *style* of the responses, in particular aspects of helpfulness and safety, as evidenced by performance in MT-Bench and AlpacaEval.
	- ((The explanation that I've heard that I like is that it helps the model elicit knowledge that is already latent within it.))
- ==[[Nectar]] Dataset==
	- The first high-quality 7-wise comparison dataset, generated through GPT-4-based ranking.
	- Authors note that high-quality RLHF datasets need all three:
		1. Diverse chat prompts
		2. High-quality and diverse responses
		3. Accurate ranking labels
	- This dataset's prompts are an amalgamation of diverse sources, including lmsys-chat-1M, [[ShareGPT]], Anthropic's [[Helpful and Harmless|HH]]-rlhf, [[UltraFeedback]], [[Evol-Instruct]], and [[FLAN v2]].
	- Responses are primarily derived from a variety of models, including GPT-4, GPT-3.5-turbo, GPT-3.5-turbo-instruct, LLaMA-2-7B-chat, Mistral-7B-Instruct.
		- The authors had a challenging time mitigating the positional bias inherent in GPT-4-based rankings. Authors conduct randomized pairwise comparisons between all response pairs before compiling a 7-wise ranking.
- Reward Model
	- We train a reward model (finetuned from LLaMA2-7B-Chat) conducting online RL based on the existing nectar dataset, using a K-wise maximum likelihood estimator under the ==Plackett-Luce Model== (as detailed in their prior paper). 
		- Discovered that for 7-wise comparisons, this new estimate yields a more effective reward model than the original loss, converting comparisons into pairwise and minimizes cross-entropy loss.
- Policy Finetuning
	- Selected [[OpenChat]] 3.5 as the initial model for policy-finetuning, owing to its high MT Bench score (7.81). 
	- Authors experimented with 1 offline RL method, [[Direct Preference Optimization|DPO]], and 3 online RL methods: [[Advantage-induced Policy Alignment]] (ADA), [[Proximal Policy Optimization]] (PPO), and [[Pairwise Proximal Policy Optimization]] (P3O).
		- DPO is simpler in implementation, and directly updates the language model on the pre-collected offline preference dataset.
		- RL methods like PPO sample new responses using the current language model, score the responses with the trained reward model, and update the LM with the reward information.
	- Authors found that the ==online RL methods yielded comparably strong results.== They found that ==DPO showed no significant improvements over the initial model Openchat 3.5==, but this was likely due to Openchat 3.5 already having done *Conditioned RL Fine-Tuning (C-RLFT)*, a different format of offline preference-based training.
	- In their current implementation of online RL methods, ==they only unfreeze the last 4 layers of the model, aiming for faster training speed.== In the future they plan to experiment with LoRA or full-parameter fine-tuning.
- Limitations
	- Starling-7B, akin to other small-sized LLMs, has its limitations. It struggles with tasks involving reasoning or mathematics and may not always accurately self-identify or ensure the factual correctness of its outputs.


Summary
> We introduce ==Starling-7B==, an open large language model (LLM) trained by ==Reinforcement Learning from AI Feedback (RLAIF).== The model harnesses the power of our new ==GPT-4 labeled ranking dataset, Nectar==, and our ==new reward training and policy tuning pipeline==. Starling-7B-alpha scores 8.09 in MT Bench with GPT-4 as a judge, ==outperforming every model to date on MT-Bench except for OpenAI’s GPT-4 and GPT-4 Turbo==. We release the ranking dataset [Nectar](https://huggingface.co/datasets/berkeley-nest/Nectar), the reward model [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha) and the language model [Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha) on HuggingFace, and an online demo in LMSYS [Chatbot Arena](https://chat.lmsys.org/). Stay tuned for our forthcoming code and paper, which will provide more details on the whole process.
> To facilitate more thorough research into RLHF, a high-quality ranking dataset specifically for chat is essential. We release ==Nectar==, a ==GPT-4 labeled ranking dataset composed of 183K chat prompts==. ==Each prompt includes 7 responses distilled from various models like GPT-4, GPT-3.5-instruct, GPT-3.5-turbo, Mistral-7B-Instruct, Llama2-7B, resulting in a total of 3.8M pairwise comparisons==. Considerable effort was invested in mitigating positional bias when prompting GPT-4 for rankings, the details of which are elaborated in the dataset section below.
> Moreover, there is a ==notable scarcity of open-source reward models==. ==We address this gap by releasing our reward model [Starling-RM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-RM-7B-alpha)==, trained with our K-wise loss on the Nectar dataset.


# Paper Figures
![[Pasted image 20240507215255.png]]

![[Pasted image 20240507220743.png]]

![[Pasted image 20240507220852.png]]

![[Pasted image 20240507221243.png]]




# Non-Paper Figures

![[Pasted image 20240418172122.png]]