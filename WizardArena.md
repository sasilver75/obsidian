---
aliases:
  - Arena Learning
---

July ~10, 2024
[[Microsoft Research]] (Luo et al.)
[Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena](https://www.microsoft.com/en-us/research/publication/arena-learning-build-data-flywheel-for-llms-post-training-via-simulated-chatbot-arena/)
#zotero 
Takeaway: Produces two artifacts: WizardLM-$\beta$ , trained using Arena Learning, and [[WizardArena]], a test set upon which performance is highly correlated with [[ChatBot Arena]]. The Arena Learning paradigm involves our model doing pairwise battle against various SoTA models with winners being judged by LLaMA3-70B-Instruct. After a set of battles, we SFT on the winning generations, and DPO/PPO on the pair of (winning, losing) generations. We then battle again.


Reference:
- Tweet: [Intro from a MSFT Wizard LM Person, mentions Wizard LM 3 is in the oven]( https://x.com/victorsungo/status/1811427047341776947)
- Blog Post: [The Rise of Agentic Data Generation by Maximme Labonne](https://huggingface.co/blog/mlabonne/agentic-datagen)

-------

## Introduction
- Building an efficient data flywheel to continuously collect feedback to improve model capabilities is a key direction for next-generation AI research.
	- LMSYS's [[ChatBot Arena]] has been a significant development in this vein, which uses a diverse set of human evaluators to compare language models in head-to-head pairwise battle, using user/judge-supplied prompts. The preference data collected here has proven to be a valuable resource for model post-training and guidance, but human-based evaluations has some challenges:
		- ==It's time-consuming and resource-intensive, which limits the scale and frequency of evaluation.==
		- ==Due to LMSYS's priority limitations, most models are unable to even participate in arena evaluations.==
		- ==The community only has access to a limited subset of conversation/preference data from LMSYS.==
- To address these challenges, this paper introduces a novel approach called [[WizardArena|Arena Learning]], which is a training and evaluation pipeline fully based on and powered by AI LLMs without human evaluators.
	- The goal is to build an efficient data flywheel and mitigate the manual and temporal costs associated with post-training LLMs.
	- The idea is that we use a powerful [[LLM-as-a-Judge]] to automatically imitate the manner of human annotators by judging a response pair from two models and provide ==scores, ranking, and explanation==.
- In the author's scenario, they simulate battles between their target model (referred to as $WizardLM-\beta$) and various SoTA models on a large scale of instruction data.
	- These "synthetic battle results" ⚔️ are used to enhance WizardLM-Beta through training strategies including [[Supervised Fine-Tuning|SFT]], [[Direct Preference Optimization|DPO]], and [[Proximal Policy Optimization|PPO]], ==enabling it to learn from the strengths and weaknesses of other good models!==
- The authors contribute a carefully-prepared offline testset called [[WizardArena]] that they hope contribute both diversity and complexity to evaluation.
	- Elo rankings produced by evaluation on WizardArena align closely with the LMSys Chatbot Arena, lending credence to the idea that WizardArena could be an alternative to human-based evaluation platforms.
- Contributions in brief:
	- [[WizardArena|Arena Learning]], a novel AI-powered method to help build an efficient data flywheel for LM post-training by simulating offline chatbot arena using an AI annotator to mitigate manual costs (time, money).
	- [[WizardArena]], a carefully-prepared offline testset. Performance on it accurately predicts Chatbot Arena Elo rankings.
	- An example of these two in conjunction to continuously improve ==WizardLM-Beta== through SFT/DPO/PPO.

## Approach
- We elaborate on the details of proposed [[WizardArena|Arena Learning]], whose closed-loop pipeline mainly contains three components:
	1. Offline Pair-wise LLM Battle Arena
	2. Iterative Post-training
	3. Model Evaluation
- Chatbot Arena and Elo rankings
	- At the core of [[ChatBot Arena]] is the use of Elo rankings, a widely-adopted system originally devised for chess players -- each model is assigned an initial arbitrary Elo ranking that's then updated after every battle based on both the outcome (win, loss, tie) and the rating difference between the competing models.
- Using a powerful LLMs as a Judge to simulate human annotators
	- At the core of simulated arena battles in Arena Learning lies a powerful LM Judge model, whose role is analyze and compare responses provided by the pair of battling models.
	- ==For the judge, they use prompt engineering with the [[LLaMA 3]]-70B-Chat model==.
	- Outputs consist of scores for each LLM, along with explanations focused on various factors, like coherence, factual accuracy, context-awareness, and overall quality, to determine whether one response is superior to the other.
	- To mitigate [[Positional Bias]], they employ a two-game setup, alternating positions of the two LLMs. Each model receives an overall scare of 1 to 10, where a higher score reflects superior overall performance.
	- This judge model is used in both Arena Learning post-training and in Wizard Arena evaluation stages.
- Build a Data Flywheel to post-train LLMs
	- Collect large-scale instruction data (I think this these are the initial instructions)
		- Arena Learning relies on a large-scale corpus of conversational data D, with the data collection involving several stages of filtering, cleaning, and deduplication to ensure the quality and diversity of the instruction data.
		- The data is split equally into some N parts for iterative training and updates $\{D_0,D_1,D_2,...,D_N\}$ 
	- Iterative battle and model evolving
		- Arena Learning employs an iterative process for training and improving the WizardLM-Beta: After each round of simulated arena battles and training data generation, WizardLM-Beta is updated using the appropriate training training strategies (SFT, DPO and/or PPO). The model is then reintroduced to the arena, where it battles against the other SoTA models once again.
		- ==As the model becomes stronger, the simulated battles become more challenging, forcing the WizardLM-Beta to push its boundaries== and learn from the latest strategies and capabilities exhibited by the other models.
		- The iterative nature also enables researchers to monitor progress and performance of the WizardLM-Beta over time, providing valuable insights into the effectiveness of different training strategies and areas for potential improvement or refinement.
- Evaluating LLMs with [[WizardArena]]
	- To accurately evaluate the performance of chatbot models and predict their Elo rankings, Arena Learning relies on a carefully-curated offline test set, designed to strike a balance between diversity and complexity.
	- Subsets of WizardArena
		- ==Diverse Subset==: Meant to capture a broad range of topics/styles/conversational contexts. To achieve this, we use SoTA embedding models to project instructions into a high-dimensional space, then cluster. Once clustering is complete, we select a representative sample from each cluster, ensuring that this subset of the test set 
		- ==Hard Subset==: Specifically designed to challenge the capabilities of even the most advanced chatbot models. We use the power of LMs to predict the difficulty level of each instruction, and we select the top-ranking samples according to the predicted difficulty scores. 
			- ((I'm curious about whether LMs are really good at predicting which instructions are actually going to be hard for them! Perhaps it depends on the prompt, we'll see if they provide that.))
	- So with the above "judge" model and hte offline WizardArena test set in place, we proceed to evaluate the performance of chatbots through pairwise battles, and the outcomes are then used to compute Elo rankings of participating chatbot models!

## Experiments
- Training data
	- Authors sample 10k [[ShareGPT]] data to train an initial model WizardLM-Beta-I_0.
	- Then, they sample instructions from open available datasets ([[Alpaca]], [[WizardLM]], [[FLAN-T5]], the HFH4 stack exchange preferences dataset, [[LMSYS-Chat-1M]], [[OpenOrca Dataset]]) and optimize them with the  following steps:
		1. Filter out illegal and toxic conversations
		2. Remove conversations with instruction lengths less than 10
		3. Eliminate duplicate instructions with prefixes of 10
		4. Employ a [[MinHash]] [[Locality Sensitive Hashing|LSH]] technique for data deduplication
		5. Used an embedding model (gte-large) to exclude instructions from the top 5 matches in semantic similarity with benchmarks (eg MT-Bench, AlpcaEval, OpenLLM leaderboard) to prevent test set leakage
		6. Removed all non-english instructions.
	- This yielded a refined ==276K instruction dataset== D, which they randomly split into 9 parts.
- Test data
	- For the WizardArena test set (Diverse and Hard subsets), they processed the source data using K-means clustering into 500 categories
	- From each category they randomly selected two samples to construct 1,000 diversity samples, named as ==Offline-Diverse WizardArena==.
	- From each category, they selected 20 samples to form a dataset of 10,000 samples, and then GPT-4-1106-preview was used to rate each instruction on a 0-10 difficulty. The top 1,000 entries were selected to create the hard test set, denoted as the ==Offline-Hard WizardArena==.
	- The ==Offline-Mix WizardArena== combines the totality of the Diverse and Hard test subsets to create 2,000 samples.
- LLM Battle
	- We select some popular models and conduct pairwise battles in the Offline-Mix WizardArena using LLaMA3-70B-Instruct as the Judge. The seem to use 100 bootstraps (pairs?) and select the median as the model ELO score.

## Ablations
- Data Selection
	- The pair-judge method used in this paper is better than selecting via random sample, kmeans cluster, instruction length, IFD, or INSTAG.
- The relationship between data size and performance
	- The size of the SFT data and gap between (chosen, reject) pairs of data is important.
	- Battle helps us filter out the truly needed data, thereby constructing a more efficient data flywheel.
- LLaMA3-Chat Judge or GPT-4 Judge?
	- People were accustomed to use GPT-4 as a judge in other papers. 
	- The [[Spearman Rank Correlation Coefficient|Spearman's Rho]] between GPT-4 judge and LLaMA3-70B-Instruct judge is 99.26%.
- Number of battle models
	- As the number of models selected for battle increases, the performance of WizardLM-Beta increases. 
- Impact of different battle modes
	- Various battle modes were attempted "We use WizardLM-β-7B-SFT-I0, Openchat-3.5, Qwen-1.5-72B, and CommandR+ as the battle group in this section"
- Performance on benchmarks
- ...
	



Abstract
> Recent work demonstrates that, post-training large language models with instruction following data have achieved colossal success. Simultaneously, human Chatbot Arena has emerged as one of the most reasonable benchmarks for model evaluation and developmental guidance. However, on the one hand, accurately selecting high-quality training sets from the constantly increasing amount of data relies heavily on intuitive experience and rough statistics. On the other hand, utilizing human annotation and evaluation of LLMs is both expensive and priority limited. To address the above challenges and build an efficient data flywheel for LLMs post-training, we propose a new method named ==Arena Learning==, by this way we can ==simulate iterative arena battles among various state-of-the-art models on a large scale of instruction data, subsequently leveraging the AI-anotated battle results to constantly enhance target model== in both supervised fine-tuning and reinforcement learning. For evaluation, we also introduce ==WizardArena==, which can ==efficiently predict accurate Elo rankings between different models based on a carefully constructed offline testset==, WizardArena ==aligns closely with the LMSYS Chatbot Arena rankings==. Experimental results demonstrate that our WizardLM-β trained with Arena Learning exhibit significant performance improvements during SFT, DPO, and PPO stages. This new fully AI-powered training and evaluation pipeline achieved 40x efficiency improvement of LLMs post-training data flywheel compare to LMSYS Chatbot Arena.


# Paper Figures
![[Pasted image 20240713135349.png|600]]
...

![[Pasted image 20240713135409.png|600]]
Given the same prompt, two models respond, and a Judge LM assigns scores to each and a rationale to determine which is better.
We can use the winning response for supervised fine-tuning, and the (winning, losing) pair for DPO or PPO preference training.

![[Pasted image 20240713142510.png|600]]
Given some training/evaluation data for the arena, have our models perform pairwise battles, in the context of an LM judge! We can use the (winning) and (winning, losing) to do both SFT and DPO/PPO, before heading back to the offline pair-wise battle arena! We can finally (or iteratively?) evaluate the perform of our model using the Wizard Arena test set.

![[Pasted image 20240713145954.png|600]]
- Train the initial version of WizardLM-Beta-SFT-I_0 on D_0
- Fight WizardLM-Beta-SFT-I_0 against SoTA LMs on D_1, and we use instances where WizardLM-B's response is the losing response, and use them to train WizardLM-Beta-SFT-I_1.
- Fight WizardLM-Beta-SFT-I_1 against SoTA LMs on D_2, and then treat the win/loss responses as the (choice, reject) to train the WizardLM-Beta-DPO-I_1.
- Fight WizardLM-Beta-DPO-I_1 and SoTA LMs on D_3, and obtain (choice, reject) pairs to train the WizardLM-Beta-PPO-I_1.

![[Pasted image 20240713155803.png|600]]
Showing the Offline-Mix WizardArena (which is just the combination of two subsets), and the categories.
I'm not sure what it means when it says that they're highly multi-turn interactions, because how do you compare a model-under-evaluation's response to a multi-turn golden response? Do you basically do like... [[Teacher Forcing]] on the model under question? What model was used for teacher responses on the WizardArena-Mix set? Or is it just whatever teacher was used for the datasets that composite the initial dataset that's filtered down to WizardArena?s

![[Pasted image 20240713175650.png|600]]
The performance of various LLMs on various categories... the whole point of this paper is that the WizardArena (Green) scores are highly-correlated with the Chatbot Arena (Red) scores -- and we can see that this is true across many models!

![[Pasted image 20240713175917.png|500]]
See that WizardArena-Mix (especially) has a higher correlation with LMSYS ChatBot Arena than other benchmarks (eg 79.36% vs 99.23%)

![[Pasted image 20240713194359.png|600]]
A great figure that shoes the LMSYS ELO compared to the various Arena ELOs (with the WizardArenaMix-ELO being the full test set). See how iterated versions of WizardLM-B improving... with the version finetuned from [[Mixtral 8x22B]] beating even [[Command R+]], which was iirc the first model to beat GPT-4-x on Chatbot Arena.

![[Pasted image 20240713200219.png|600]]
Showing the improvement in ELO scores across incarnations of the model (SFT1, DPO1, PPO1, SFT2, ...)

![[Pasted image 20240713202606.png|600]]
Exploring some data selection strategies that could be used during the first round of SFT.
It seems like the INSTAG method is pretty good as a comparison point -- we'll look at that in a different paper!

# Non-Paper Figures














