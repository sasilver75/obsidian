October 2, 2023 (5 months after [[UltraChat]])
Tsinghua University et others (Cui et al.) - Only 1 author overlap with [[UltraFeedback]]
[UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377)
#zotero 
Takeaway: A large-scale, fine-grained, diverse, synthetic preference dataset with instructions sampled from a pool of open preference datasets, and responses sampled from a variety of closed and open models. Annotations are provided by [[GPT-4]] and include both rationales *and* fine-grained scalar scores across the *aspects* of Instruction Following, Helpfulness, Honesty, Truthfulness. Each instruction is responded to by four LMs sampled from our pool, with each LM prompted along with a single *principle* related to an *aspect* associated with the dataset from which the current prompt was sampled. Authors train a Reward Model (UltraRM), Critique Model (UltraCM), and Language Model (UltraLM). We collect about 64k prompts from diverse sources ([[UltraChat]], [[ShareGPT]], [[Evol-Instruct]], [[TruthfulQA]], [[FalseQA]], [[FLAN v2|FLAN Collection]]) and generate 4 responses for each prompt, resulting in 256k samples (you could interpret this as ~1M binarized comparison pairs).

HuggingFace Dataset Card: [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)

Note: When training [Notus 7B](https://huggingface.co/argilla/notus-7b-v1) (an "improvement" vs. [[Zephyr]]-Beta), the authors from [[Argilla]] noticed some issues in the original UltraFeedback dataset, leading to high-scores for bad responses... they manually-curated several hundreds of data points, and then binarized the dataset (for [[Direct Preference Optimization|DPO]]) and verified it with the Argilla platform... It led to a new dataset where the chosen response is different in ~50% of cases! This dataset is named [ultrafeedback-binarized-preferences](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) and is available on the hub.

---
## Introduction
- Relying solely on supervised imitation learning in training leads to well-known issues -- learning from human feedback is critical!
- Human data annotators are often assumed by default to be human beings who can provide flexible and accurate supervision signals, yet the data they generate is severely bounded by factors like financial resources, time, and knowledge.
- This study aims to scale feedback in an efficient manner with *AI feedback,* which substitutes human annotators with advanced LLMs.
	- It's more scalable -- it's possible to collect more, at a lower cost.
	- Its quality improves as the LLM annotators become more capable.
- Besides scalability, we prioritize *diversity* of both instructions and responses for holistic language model alignment.
	- We compile a diverse array of ==over 60,000 instructions and 17 models from multiple sources to produce comparative conversations in broad topics and quality.== 
- We use ==techniques to alleviate annotation bias and improve feedback quality:==
	1. Decomposing annotation documents into four different aspects, namely *instruction-following, truthfulness, honesty, and helpfulness*, to reduce ambiguity.
	2. Providing *objective grading criteria and reference responses* for score calibration.
	3. Asking GPT-4 for detailed textual critique *before* scores as chain-of-thought rationales.
		- ((Thank god! Some papers provide a rating before an explanation, which I don't understand.))
- Authors use [[UltraFeedback]] to train ==UltraRM==, an open-source state of the art reward model. They also train ==UltraCM== (critique model) off of the *rationales* in the UltraFeedback dataset. They then use UltraRM to train ==UltraLM,== a finetune of LLaMA2-13B using [[Best-of-N Sampling]] and [[Proximal Policy Optimization|PPO]].

## UltraFeedback
- We take into account *scalability* and *diversity* in all three stages of the preference data collection process:
	1. Collecting instructions
	2. Sampling completions
	3. Annotating comparison pairs
- We collect a large-scale and diversified instruction set hoping to enhance four aspects of LM capabilities:
	1. Instruction Following
		- Randomly sample 10k from both [[Evol-Instruct]] and [[UltraChat]], and 20k from [[ShareGPT]]
	2. Helpfulness and Informativeness (useful and correct answers)
		- Randomly sample 10k from both [[Evol-Instruct]] and [[UltraChat]], and 20k from [[ShareGPT]]
		- Also include [[FLAN v2|FLAN Collection]] because of the great task diversity within FLAN. We adopt a stratified sampling strategy, randomly picking 3k instructions from the "CoT" subset and sampling 10 instructions per task for the other three subsets, while excluding those with overly long instructions, while excluding those with overly=long instructions.
	3. Truthfulness (groundedness, avoiding self-contradiction)
		- All instructions from the [[TruthfulQA]] and [[FalseQA]]
	4. Honesty (Know what they (don't) know and express uncertainty towards given problems)
		- Assessed by [[TruthfulQA]] and [[FLAN v2|FLAN Collection]], since they both contain reference answers, so it's easier for the annotator to judge if the uncertainty expressed in LM responses calibrates with the accuracy.
- To guarantee that the collected responses are dissimilar and well-distributed, they use *different models* to generate completions for each instruction -- both different of series of models and different model sizes within model families.
	- Including GPT-4. GPT-3.5-Turbo, Bad, UltraLM-13B/65B, WizardLM-7Bv1.1./13B-v1.2/70B-v1.1, Vicuna-33B-v1.3, LLaMA2-7B/13B/70B-Chat, Alpaca-7B, MPT-30B-Chat, Falcon-40B-Instruct, Star-Chat, Pythia-12B.
	- ==Authors randomly sample four different models from the above pool to complete each instruction. ==
	- To *further improve diversity*, authors elicit distinct model behaviors by adding *different principles* before completing each instruction.
		- ==They hand-craft one *principle* for each of the four aspects (InFo, Helpfulness, Truthfulness, Honesty), and then use GPT-4 to curate another ten based on the human-written example.==
		- "According to dataset characteristics, each data source is assigned with different principle prompts."
		- ==Authors randomly sample a corresponding principle for each completion and add it to the system prompt to induce model behaviors.==
- After generating 255,864 model completions based on 63,967 instructions, we employ [[GPT-4]] to provide two types of feedback for each completion
	1. Scalar scores indicating fine-grained quality regarding multiple aspects (InFo, Helpfulness, Truthfulness, Honesty).
	2. Textual critique that gives detailed  guidance on how to *improve* the completion.
- Preference Annotation
	- Regarding potential subjectivity and randomness of GPT-4 annotation, we apply four techniques to improve annotation quality:
		1. ==Decomposition==: To reduce ambiguity and difficulty of annotation, we compose overall quality assessment into four fine-grained assessments (InFo, Helpfulness, Truthfulness, Honesty)
		2. ==Standard==: For each aspect, we provide GPT-4 with detailed exemplars of scores from 1 to 5 for reference, to try to avoid subjective standards.
		3. ==Reference==: To prevent inconsistency ratings across different runs, we wrap one instruction and all of its (4) completions into the prompt and ask GPT-4 to score four completions simultaneously to reduce randomness.
		4. ==Rationale==: Besides scoring each response, GPT-4 is required to generate a rationale on how the response should be scored, according to the documentation. 
- Now we have four fine-grained scalar scores and rationales for each response.
- UltraFeedback stands out to be one of the largest of the available preference and critique datasets... and it's the only dataset (they compared to) that provides *both* scalar preferences *and* textual feedback, letting it serve a preference dataset (like [[oasst1]]) *and* a critique dataset (like [[Shepherd]])
- Authors develop ==UltraRM==, an advanced open-source reward model that provides preferences for AI responses given user instructions, and a critique model ==UltraCM== from the textual feedback/rationales in UltraFeedback. 
	- UltraRM is based on LLaMA2-13B. They mix several open-source datasets ([[Stanford Human Preferences]], OpenAI Summarization, [[Helpful and Harmless|HH]]) to train it.
	- UltraCM is a critique model with the same initialization as UltraRM, but is trained solely on UltraFeedback *critique data* -- the 255,864 in total.


## Experiments
- Experiments include:
	- Evaluating UltraRM on human preference benchmarks
	- Test UltraRM in enhancing chat language models with two strategies: Best-of-N sampling and reinforcement learning.
	- We evaluate the feedback quality of UltraCM.
- Reward Modeling
	- To see how UltraRM aligns with human preference, they conduct experiments on four human-annotated preference datasets:
		- OpenAI WebGPT
		- OpenAI Summarization
		- Anthropic HH-RLHF
		- Stanford SHP
	- On each dataset, they calculate the rewards of two responses for one prompt and predict which one is more preferred, and compare to the human preference dataset labels... It seems that it gets an average of 71.0% accuracy across the four datasets, which doesn't seem *that* impressive 🤔. It seems to be nearly on par with the "LLaMA2 Helpfulness" model (which is a closed model).  
		- ((Note that this model gets something like a 67 on [[RewardBench]], and better models like Nemotron or even some 8B LLaMA3 finetunes get like 90+; so this is no longer SoTA))
- Best of N Experiments
	- To verify that UltraRM could serve as a good indicator of response quality, they conduct a best-of-n experiment: They randomly sample 16 completions from the original UltraLM-13B on the [[AlpacaEval]] benchmark, and select the best response by the reward model. While we can get an initial one-sample 76.53% win rate against AlpacaEval's text-davinci-003 baseline, by increasing to best-of-16, our win rate hits 91.54%. See figure.
- PPO Experiment
	- We use UltraRM in an [[Reinforcement Learning from from AI Feedback|RLAIF]] experiment with [[Proximal Policy Optimization|PPO]], tuning it for 80 iterations on the UltraFeedback prompt (512 samples/iteration). UltraLM-13B-PPO overtakes LLAMA-based models, even beating the much larger LLaMA2-70B-Chat IFT'd model.

## Agreement with Human Preferences and Analysis
- We randomly sample 400 comparison pairs from UltraFeedback/AlpacaEval/Evol-Instruct/UltraChat test sets (100 each) and ask 3 independent annotators to compare those pairs (win/tie/lose).
- We present the agreement ratio between GPT-4 and the Annotators, which shows that the average inter-annotator agreement is something like 56-60%, with GPT-4 actually being the one with the highest 60% agreement with the annotators.

## Related Work
- Incorporating human feedback with imitation learning and RL has been the mainstream approach to alignment.
- [[Scalable Oversight]] aims to supervise potent AI models by models themselves, as models approach superhuman capability (or at least above the capabilities of annotators).
- [[Constitutional AI]] to let LLMs refine their responses given a set of regulations.


## Conclusion



Abstract
> Reinforcement learning from human feedback (RLHF) has become a pivot technique in aligning large language models (LLMs) with human preferences. ==In RLHF practice, preference data plays a crucial role== in bridging human proclivity and LLMs. However, the ==scarcity of diverse, naturalistic datasets of human preferences== on LLM outputs at scale poses a great challenge to RLHF as well as feedback learning research within the open-source community. ==Current preference datasets, either proprietary or limited in size and prompt variety,== result in limited RLHF adoption in open-source models and hinder further exploration. In this study, we propose ==ULTRAFEEDBACK==, a ==large-scale, high-quality, and diversified preference dataset designed to overcome these limitations and foster RLHF development==. To create ULTRAFEEDBACK, we compile a diverse array of instructions and models from multiple sources to produce comparative data. We meticulously devise annotation instructions and ==employ GPT-4 to offer detailed feedback in both numerical and textual forms==. ULTRAFEEDBACK establishes a reproducible and expandable preference data construction pipeline, serving as a solid foundation for future RLHF and feedback learning research. Utilizing ULTRAFEEDBACK, we train various models to demonstrate its effectiveness, including the reward model UltraRM, chat language model UltraLM-13B-PPO, and critique model UltraCM. Experimental results indicate that our models outperform existing open-source models, achieving top performance across multiple benchmarks. Our data and models are available at [this https URL](https://github.com/thunlp/UltraFeedback).

# Paper Figures
![[Pasted image 20240721001110.png]]
The bit about "sampling instructions...from large pools" to guarantee diversity bit doesn't make a lot of sense to me... It assumes that these pools have all the diversity you need, which... might be right?

![[Pasted image 20240721003950.png|600]]
Comparison with some other preference and critique datasets including [[oasst1]], [[Helpful and Harmless|HH]], [[Shepherd]], and others.

![[Pasted image 20240721011105.png|500]]
UltraRM accuracy on out-of-sample *real* human preference datasets. The 71.0 doesn't seem that impressive, huh? Note that this is just a 13B model. On [RewardBench](https://huggingface.co/spaces/allenai/reward-bench), it gets a 67.6 overall score, compared to (eg) 92.2 for [[Nemotron-4]]'s 340B Reward.

![[Pasted image 20240721014035.png|400]]
From the Best-of-N Experiments; Plots the win rate against text-davinci-003 on [[AlpacaEval]]. We sample *n* responses and choose the one with the highest reawrd.

![[Pasted image 20240721014231.png|600]]

![[Pasted image 20240721015126.png|300]]

![[Pasted image 20240721021752.png|500]]

![[Pasted image 20240721021834.png|600]]

![[Pasted image 20240721022236.png|500]]

![[Pasted image 20240721022541.png|500]]

![[Pasted image 20240721022656.png|500]]

![[Pasted image 20240721022754.png|500]]

![[Pasted image 20240721022812.png|500]]

![[Pasted image 20240721022823.png|500]]



# Non-Paper Figures
![[Pasted image 20240424153738.png|300]]

![[Pasted image 20240721010554.png|300]]
Dataset breakdown from HF card


![[Pasted image 20240717105802.png|400]]
An example of one poorly-provided human feedback in the UltraChat dataset noticed by Argilla when training Notus.