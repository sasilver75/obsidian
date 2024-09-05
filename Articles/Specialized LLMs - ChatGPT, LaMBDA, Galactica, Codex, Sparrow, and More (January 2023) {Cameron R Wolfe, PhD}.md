#article 
Link: https://cameronrwolfe.substack.com/p/specialized-llms-chatgpt-lamda-galactica

-----

In this overview, we'll explore methods of replacing and improving LLMs for a variety of use-cases.

We can modify the behavior of LLMs by using techniques like:
1. Domain-specific pretraining (eg BloombergGPT)
2. [[Supervised Fine-Tuning]]
3. Model alignment (eg via [[Reinforcement Learning from Human Feedback|RLHF]])

These methods can be used to combat known limitations of LLMs and modify them to better suit our needs, through either/both behavior and knowledge.

(Skipping some descriptions of "what are language models," etc.)

# Publications
Now let's overview a variety of publications that extend generic LLMs to more specialized scenarios. Numerous different methodologies are used to modify and improve LLMs, but the general concept is the same.

## Evaluating Large Language Models Trained on Code (July 7, 2021)
- This paper introduces [[Codex]], a finetune of a GPT model on code data from GitHub.
- ==Given a Python docstring, Codex is tasked with generating a working Python function that performs the task outlined in the docstring.==
![[Pasted image 20240501192220.png|300]]
- Codex is quite a bit smaller than GPT-3 at only ==12B parameters==, finetuned over a 159Gb corpus of Python files from GitHub.
- Authors create the [[HumanEval]] dataset to evaluate Codex, which is a set of 164 programming problems with associated unit tests. The model is evaluated on its ability to write code that passes the test, given a certain number of attempts (==pass@k==).

## LaMBDA: Language Modeling for Dialog Applications (Jan 20, 2022)
- In DeepMind's [[LaMDA]] model (Jan 20, 2022), authors  create a LLM-powered dialog model, with the largest of those created being 137B parameters, slightly smaller than GPT-3.
- The authors define three important areas of alignment for LLM behavior: Quality, Safety, Groundedness.
- The authors use a human workforce to collect/annotate examples of model behavior that *violates guidelines* -- and uses it to finetune Lambda in a supervised manner in some way (Assumedly one that makes it *not* do these generations).
- We see that large-scale pretraining of LLMs might not be ALL that's required to make LLMs as useful as possible, especially when adapting them to domains like dialog generation -- finetuning datasets are important, but expensive.

## Training Language Models to Follow Instructions with Human Feedback (March 4, 2022)
- We continue the trend of aligning LLM behavior based on human feedback, but instead we use an [[Reinforcement Learning from Human Feedback|RLHF]] approach to train a model called [[InstructGPT]] to produce outputs that are aligned with human preferences.
- Beginning with a GPT-3 models of varying sizes (1.3B, 6B, 175B), the alignment process proceeds in three phases:
	- Supervised Finetuning/Instruction-Tuning
	- Reward Model (RM) Training (6B parameter RM)
		- The reward model is trained over pairs of model responses that have been assigned human preference score. 
	- RLHF
		- We use the RM's output as a scalar reward to optimize our original LLM using the [[Proximal Policy Optimization|PPO]] algorithm.
- Makes note of a slight [[Alignment Tax]] in evaluations on public datasets, but authors show that these can be minimized by mixing in standard language model pre-training updates during the alignment process.
- The insights from InstructGPT were critical in scaling up to [[ChatGPT]] ~7 months later.

## Improving Alignment of Dialogue Agents via Targeted Human Judgements (Sep 28, 2022)
- The [[Sparrow]] paper from [[DeepMind]] can participate in information-seeking dialogue (i.e. dialog focused on providing answers and follow-ups to questions with humans) with humans, and even support its factual claims with information from the internet.
- Sparrow is initialized using 70B parameter [[Chinchilla]] model, a generic LLM pretrained over a large text corpus.
- Authors then use RLHF to align the LLM to their desired behavior, as well as enable the model to search the *internet* for evidence of factual claims!
	- This is done by injecting extra "participants" into the dialog, called "Search Query" and "Search Result". These are just another sequence of tokens that the mode can generate -- we have to teach Sparrow how to do this in a supervised fashion during the alignment process.
- Sparrow uses RLHF for alignment
	- Authors define an itemized set of rules that characterize model behavior according to their alignment principles of helpful, correct, and harmless.
	- These rules are for *==human annotators==* to better characterize model failures and provide targeted feedback at specific problems.
- Human feedback is collected using:
	1. ==Per-turn Response Preference==
		- Per-turn response preferences provide humans with an incomplete dialog and multiple potential responses that complete the dialog. Humans then identify the response that they prefer. ((This seems like the usual human preference gathering for RLHF))
	2. ==[[Adversarial Probing]]==
		- Humans are asked to:
			- Focus on a single rule
			- Try to elicit a violation of this rule by the model
			- Identify whether the rule was violated or not
- ==Separate reward models== are trined on the per-turn response and rule-violation data.
	- We then fine-tune the model in RL based on both of these reward models.
	- We can also repurpose the two reward models to rank potential responses generated by Sparrow. To do this, we simply generate several responses and choose the ones with _(i)_ ==the highest preference score from our preference reward model and _(ii)_ the lowest likelihood of violating a rule based on our rule reward model==. However, ranking outputs in this way does make inference more computationally expensive.
- To ensure that sparrow learns to *search* for relevant information, response preferences are always collected using *four options
	- Two options contain no evidence within the response
	- Others must generate a search query, condition upon the search results, then generate a final response.
- Authors also use some sort of form of [[Self-Play]] in this paper.
- "When the resulting model is evaluated, users prefer this model's output related to several baselines, including LLMs that undergo SFT over dialog-specific datasets."
	- ((Does this mean that Sparrow isn't SFT'd?))

## Galactica: A Large Language Model for Science  (Nov 16, 2022)
- [[Meta AI Research]]'s [[Galactica]] model is a model meant to store, combine, and reason about scientific knowledge from several fields. 
- It's pretrained using a LM objective on a bunch of scientific content, including 48M papers, textbooks, lecture notes, and more specialized databases (eg known compounds and proteins, scientific websites, encyclopedias, etc.)
- ==Unlike most LLMs, Galactica is pre-trained using a smaller, high-quality domain-specific scientific corpus.==
- Galactica adopts a ==special tokenization procedure== so that the data ingested by the model is still contextual (for DNA sequences, latex, chemical compounds). Galactica uses special tokens to identify scientific citations and portions of the model's input/output to which step-by-step reasoning should be applied.
- 125M-120B parameter models
- After pretraining, the model is finetuned over a dataset of prompts. 
- ==DRAMA==
	- Galactica was released by Meta with a public demo; shortly after its release, there was a ton of backlash from the research community, because Galactica can generate reasonable-sounding scientific information that's potentially incorrect.













