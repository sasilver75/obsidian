#article  #premium
Link: https://gaotianyu.xyz/blog/2023/11/30/instruction-tuning/

Data, algorithms, and eva;iatopm on finetuning

-------

Disclaimer: This isn't a comprehensive review of instruction-tuning or RLHF literature -- just a brief introduction to some of the recent progress and a little self-promotion.

---

Large language models are powered by billions of parameters and trained with trillions of tokens -- but to be useful in real-world applications and to act as a general task-solving machine, they must master following user instructions and responding in a coherent and helpful way instead of being a mere "stochastic parrot."

Thus, open-ended instruction tuning ([[InstructGPT]]), fine-tuning an LLM such that it can follow user instructions and respond in a helpful and harmless way ([[Helpful and Harmless|HH]]).

The interest increased further after the huge success of ChatGPT

Open-ended instruction tuning usually contains two stages:

1. Supervised Fine-Tuning ([[Supervised Fine-Tuning|SFT]]): The model on collected use instructions and gold responses.
2. Alignment: Most commonly [[Reinforcement Learning from Human Feedback|RLHF]] , the model is aligned with human preferences; usually requires human preference data, labeled by humans.

Let's cover recent efforts in four parts ==((I think this is a great way to discuss recent alignment work!))==:
1. SFT data
2. Preference data
3. Algorithms
4. Evaluation


# Supervised Fine-Tuning (SFT) Data
- There are generally two purposes of SFT, corresponding to two types of data:
	1. To improve the general language understanding abilities of LLMs, reflected in benchmarks like [[HellaSWAG]] and [[Massive Multi-task Language Understanding|MMLU]]
		- ==There are multi-task instruction-tuning datasets, which have been heavily explored between 2020-2022. These data combine thousands of NLP tasks together, so that one can train models on the combination in a multi-task way.==
		- [[Natural Instructions]], [[Super-NaturalInstructions]], T0 (San), and [[FLAN]] (dataset)
		- These datasets/models target traditional NLP tasks (QA, NLI) and then to have shorter/simpler/less-diverse instructions and responses. "Classify the sentiment of this sentence," not "Write me a personal webpage in a similar style to OpenAI's blog using Jekyll". 
		- [[Tulu]] showed us that combining these datasets with open-ended instruction-tuning datasets can improve both the general language ability and instruction-following ability.
		- [[Orca]] found that using these data as seeds, prompting GPT-4 to output answers with explanations, and imitating GPT-4's responses can significantly improve weaker models' performance.
	2. To train LLMs to follow instructions, acquire conversational abilities, and to be helpful and harmless. Notably emerged in 2023.
		- ==The general idea is that training with these data doesn't improve LLMs' "knowledge", but merely "guides" them to follow the instruction-following or conversational format, gaining an engaging tone, being polite, etc.==
		- Collecting SFT is expensive, as one needs to collect both user instructions as well as annotate the demonstration responses.
		- One primary way for open-source LLMs to get open-ended instruction-tuning data is to distill from proprietary LLMs.
			- One of the earliest open-source instruction models, [[Alpaca]], used [[Self-Instruct]] to prompt `text-davinci-003` to generate pseudo-SFT data, and then SFT'd LLaMA-7B on it.
			- [[WizardLM]] also improved data diversity by using ChatGPT to rewrite Alpaca data iteratively.
			- [[UltraChat]] first constructed questions automatically using different strategies, then prompted ChatGPT to simulate a conversation given the question.
			- [[Vicuna]] and [[Koala]] explored [[ShareGPT]] data as SFT data.
			- A recent similar effort, WildChat, provided online users free ChatGPT access and collected the conversations, though the focus was more on studying toxic use cases.
		- ==Even though it's a relatively cheap way to acquire data, imitating proprietary LLMs is found to just "mimic ChatGPT's style, but not its Functionality," putting a question on how far open-source models can really go by solely relying on such SFT data.==
		- Another way to collect SFT data is to manually annotate a small amount of data
			- Open Assistant initiated a crowd-sourcing effort where volunteers write both instructions and responses ([[oasst1]], [[oasst2]]).
			- [[Dolly]] contains 15k Databricks-employee-generated data data (more towards Wikipedia-based factoid question-answering)
			- [[LIMA]] is a collection of author-curated 1,000 SFT data (has a distribution heavily-steered towards Stack Exchange and Wiki How, and is found to be surprisingly effective in producing strong instruction models)


We see two promises with the above methods:
1. LLaMA-2-70B-chat, a LLaMA-70B model tuned on closed-source data, is shown to be *more helpful* than ChatGPT by human evaluation. This shows that we have a base model that is potentially as strong as ChatGPT's base model to play with!
2. Our research community has already conducted exciting research on the "toy" or "laboratory" data, such as better alignment algorithms, which will be mentioned below.


---

# What about preference data?
- Despite impressive "illusion" that open-source SFT models give us (in fact, this illusion ignited the trend of open-source efforts in instruction tuning), merely having SFT is not enough!
- Aligning the model with human preference data is essential for models to be better language assistants. 
- Think about how we want models to be "honest" -- SFT almost *always* encourages the model to *given an answer*, and hardly ever teaches the LM to say "I don't know about this."
- Alignment algorithms have been shown to bring better "human satisfaction" in several works, but most open-source models haven't yet gone through the alignment stage (RLHF) due to
	1. High costs to run
	2. Brittleness to tune [[Proximal Policy Optimization|PPO]]
	3. Lack of high-quality preference data (this also hinders the development of good new algorithms)

The most commonly used two preference datasets for developing alignment algorithms algorithms are:
- [[OpenAI]]'s [[TL;DR]] preference data re: summarizations
- [[Anthropic]]'s [[Helpful and Harmless|HH]]-RLHF dataset, human-model open-ended dialogues
==Both of these have good qualities, but the diversity and complexity of instructions are not comparable to nowaday's SFT data.==

==In 2003, there emerged a number of new preference datasets==

There are ==crowd-sourcing efforts== to collect preferences from humans
- OpenAssistant
- [[ChatBot Arena]]

More datasets take a simulated or ==heuristic approach to targeting existing data==: [[Stanford Human Preferences|SHP]] uses numbers-of-upvote heuristics on Reddit to construct a synthetic preference dataset.

Others use GPT-4 as a gold annotator ([[AlpacaFarm]], [[UltraFeedback]], others).

HuggingFace recently released [[Zephyr]], trained with [[UltraChat]] for SFT and [[UltraFeedback]] for alignment (using [[Direct Preference Optimization|DPO]]), which is shown to have comparable performance to LLaMA-2-Chat-70B, a model trained on closed-source data!

Another line of work tries to use "AI feedback" using LLMs to guide LLMs without human involvement. The idea is different from "using GPT-4 as an annotator", since GPT-4 is still trained with human preferences, but the here the goal is for the model to bootstrap *without* human preference data.
- [[Constitutional AI]]: A series of principles that good generations should follow, and prompts an SFT model to self-improve its generations (by self-critiques and subsequent revision), and then fine-tunes the model on the self-improved generation. They also use [[Reinforcement Learning from from AI Feedback|RLAIF]], where one prompts the SFT model to generate preferences over output pairs (instead of using a human).
	- This demonstrates that starting from a model trained on only "helpfulness" human supervision, it's then possible to train the model to be "harmless" (without human supervision)


# Is RL the only way?
- Using PPO for RLHF has been the primary method for alignment (for example, it was used in [[InstructGPT]], [[ChatGPT]], [[GPT-4]], and [[LLaMA 2]] Chat).
	- The basic idea is that you first train a reward model on preference data, then you can use the reward model to provide feedback, using RL to tune the model.
- RLHF is effective, but complicated to implement, prone to optimization stability, and sensitive to hyperparameters. There are a number of new methods proposed that can align models with preference data, and some even claim to be stronger than RLHF!

==Best-of-N==
- Intuition: The model, after SFT, already has the potential to generate good outputs -- we just need to pick them out!
- In [[WebGPT]] and the Summarization from Human feedback paper, authors explored "best-of-n" sampling -- ==sampling n outputs and using a reward model to pick the best.== 
	- However, ==Best-of-N== is inefficient if the final optimal policy is very far away from the original SFT model (n increases exponentially to the KL between final policy and the SFT model), not to mention that even if n is small, it is very inefficient to run inference for all of these n responses.

==Expert Iteration== ((?))
- There are some methods that use best-of-n training -- we can sample a lot during training, pick the best ones, and SFT on them. This can be combined with ==sampling the best-of-n online (sample n outputs, pick the best one, train on the best one, repeat), which is essentially "expert iteration"== (Anthony et al, 2017); the sampling of best-of-n can also be combined with "natural language feedback."

==Conditional Tokens==
- Another idea is of "conditional tokens": One can do SFT on LMs with both good and bad examples, and ==prepend a "good" prompt to the good examples, and a "Bad" prompt to the bad examples==. During inference, you can then condition the model with the "good" prefix and expect the model to generate good outputs.
	- ((Interesting that we teach it both what good AND bad examples look like.))

==Contrastive-based methods==
- There are several newly-proposed methods that closely-resemble the idea of contrastive learning: ==One can get the probabilities of both good and bad examples from the model, and "promote" the good ones while "repressing" the bad ones.==
- SLiC (2022, 2023) and RRHF (2023) both optimize a contrastive ranking loss and a regularization loss.

SLiC optimization:
![[Pasted image 20240422225124.png]]
Where $\pi_{\theta}$ is the language model, and $x, y_w, y_l$ and $y_{ref}$   are the instruction, winning output, losing output, and reference output.

PRO (2023) adopts a softmax-form of contrastive loss and optimizes over multiple negative outputs instead of just one.

[[DPO]] takes on a similar idea, but starts from the RLHF objective function.
![[Pasted image 20240422225602.png]]


==One downside of these models is that they either sample $y_w$ or $y_l$ from the SFT model or take them directly from existing datasets (thus, sampling from other models), creating a distribution mismatch.==

Liu et al (2023) in RSO propose to fix this problem with sampling from the optimal policy $\pi^*$  -- by doing [[Rejection Sampling]] with the reward model. They showed that by applying such a sampling strategy on top of SLiC or DPO can improve the final model performance.

These methods have received much attention recently and demonstrated that DPO can bring significant improvements over SFT.

HuggingFace's [[Zephyr]] model, also trained with DPO, achieves strong performance on MT-Bench, even comparable to LLaMA-2-chat and GPT-3.5. These methods are much cheaper than RL, so it's good news to the research and open-source community, and can potentially inspire better alignment algorithms.

We still need to better understand the properties of models trained with alignment algorithms, and whether they truly help them learn useful figures.
- Authors have shown that on several popular datasets, the learned reward models often have preferences *highly correlated* to length.


# Evaluation
- Human evaluation remains the "gold standard" for assessing the abilities of open-ended conversational models -- but human evaluation remains ==unreliable== (especially if one uses cheap crowdsourcing platforms like Amazon Mechanical Turk). Humans are also ==costly==!
- As a result, people have started to use stronger LLMs to evaluate weaker LLMs (eg ChatGPT to evaluate LLaMA-based models).
	- As long as there is such a big ability gap, models like GPT-4 should suffice as evaluators.

Several pilot works using LLMs as evaluators demonstrated reassuring results -- ==LLM evaluators often have strong agreement with human evaluation.== 

On the other hand, there are a number of papers that show ==LLM evaluators being extremely sensitive to certain biases== -- for example, swapping their preferences if you swap the two outputs to be compared. They also favor long outputs and outputs generated by a similar model. 
- As a result, there are several "meta-evaluation" benchmarks proposed to evaluate how good LLM evaluators are (usually in the form of accuracy on human preference data), namely FairEval, MT-Bench, and LLMEval^2.
- ==The human annotations of these benchmarks are often noisy and subjective, and intrinsic human agreement rate is quite low (AlpacaFarm reports 66%, MTBench reports 63%, and FairEval reports 71.7%.==). So it's unclear then, whether we can trust those meta-evaluation benchmarks, and the LLM evaluators.



(Introducing the Author's work!)

### LLMBar: A Better Meta-Evaluation of LLM evaluators
- In recent work in ==Evaluating Large Language Models at Evaluating Instruction Following==, we rethink the problem of meta-evaluation. We argue that previous works ignore one important factor -- the intrinsic subjectivity of human preferences!
![[Pasted image 20240422231723.png]]
The quality difference between these two is discernable; human annotators prefer the longer one, adding this bias to the preference dataset.
When we assess LLM evaluators based on such subjective and noisy meta benchmarks, ==we can't guarantee that the high-scoring evaluators can reliably evaluate objective properties, like instruction-following or factual correctness, over subjective preferences like output length==.

Following this path, we create a new meta-evaluation benchmark, LLMBar, that focuses on one objective criterion -- **==instruction following==**!
We chose this because: 
1. it can be objectively evaluated
2. It is directly related to desirable LLM properties like *helpfulness*
3. Unlike superficial qualities that can be easily acquired via imitation learning, even the strongest LLMs today struggle on this benchmark
	- ((If this is a meta-benchmark for LLM evaluators of LLM outputs, how do we create a good LLM evaluator of instruction following when even the strongest LLMs struggle to instruction-follow, in some cases?))

![[Pasted image 20240422232558.png]]

==Even though it's clear that the right output follow the instruction better than the left one, both human and LLM evaluators often (incorrectly) prefer the left one due to its engaging tone!==

In LLMBar, the authors manually curate 419 instances, where each entry consists of an instruction with two outputs:
- One faithfully follows the instruction, and the other deviates, and there's always an objective preference.
- ==LLMBar has a human agreement rate of 94%, thanks to the objective criterion and manual curation.==

We test the evaluators on those output pairs and compare the evaluator preferences to our gold labels. We also ==curate an adversarial set==, where the "bad" output often has some superficial appeal (length, engaging tones, generated by a better LM, etc.)

![[Pasted image 20240422233941.png]]
Interesting how LLaMA2 falls off!

Besides different LLMs, we also show that different prompts matter a lot for the evaluator. Several previous works explored in this direction: [Wang et al., 2023](https://arxiv.org/abs/2305.17926) proposed sampling multiple explanations and aggregating them into a final judgment; [Zheng et al., 2023](https://arxiv.org/abs/2306.05685) suggested a reference-guided method, where the LLM evaluator first generates its own output given the instruction, and then uses it as a reference; there are also several papers showing that deploying multiple evaluators (different LLMs or prompts) and letting them communicate or synthesize their judgements can improve the evaluator accuracy ([Li et al., 2023](https://arxiv.org/abs/2307.02762); [Zhang et al., 2023](https://arxiv.org/pdf/2308.01862.pdf); [Chan et al., 2023](https://arxiv.org/abs/2308.07201)).

In our work, we propose a combo of methods: **metrics+reference+rules** (as shown below). We first prompt the LLM to generate three instruction-specific metrics or rubrics (a recent work, [Saha et al., 2023](https://arxiv.org/abs/2310.15123), proposed a similar strategy); we also prompt the LLM to generate a reference output. Then, we feed the LLM the metrics and the reference, explicitly list the rules (e.g., focusing on instruction following, ignoring positional bias), and ask the model to give a judgement. Compared to a vanilla prompt used in [AlpacaFarm](https://arxiv.org/abs/2305.14387), our new prompt significantly improves the evaluator performance on LLMBar (10% boost for GPT-4 on the adversarial set). We have more ablation studies in the paper and more interesting results, for example, chain of thought ([Wei et al., 2023](https://arxiv.org/abs/2201.11903)) hurts the evaluator accuracy most of the time, a counter-intuitive finding.

![[Pasted image 20240422234034.png]]



# Looking forward

The emergence of open-source instruction-tuning data, algorithms, and models is one of the most exciting progress for LLMs in 2023. It gives researchers and developers the chance to train, evaluate, interact, and analyze instruction models with full control (from parameters to data), which only existed as black boxes before. The past few months are also a bit “chaotic” for the field, as hundreds of papers released results with different data, algorithms, base models, and even evaluation, making it hard to cross-compare the literature. I expect that the community will soon converge to some standard data/evaluation and we can develop better instruction-tuning models in a more scientific and reproducible way!




















