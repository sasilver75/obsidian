#article 
Link: https://cameronrwolfe.substack.com/p/rlaif-reinforcement-learning-from

----------

Beyond using larger models and datasets for pretraining, the drastic increase in LLM quality has been due to advancements in the alignment process, which is largely been fueld by finetuning techniques like:
- [[Supervised Fine-Tuning]] (SFT)
- [[Reinforcement Learning from Human Feedback]] (RLHF)
	- A particularly interesting technique that lets us directly finetune a language model based on human-provided preferences.

RLHF allowing us to teach a LM to produce human-satisfying content is great, but it requires that a large amount of human preference labels be collected, which can be expensive and time consuming!
- How can we reduce the onerous cost of human preference collection, when it comes to RLHF? Enter: [[Reinforcement Learning from from AI Feedback]]

---

# Background Information

... Skipping background information on language models, SFT, RLHF ...

# Automating RLHF with AI Feedback
- Despite its effectiveness, RLHF requires a lot of human preference annotations to work well.
	- [[LLaMA 2]] for example is trained using >1M human preference annotations!
- Recent research has found that LLMs can generate accurate preference labels if prompted correctly -- a variety of papers have explored this topic.


## Training a Helpful and Harmless Assistant with RLHF \[8\] (Anthropic, Apr 2022)
- Link: https://arxiv.org/abs/2204.05862
- Authors trained a language model to be helpful and harmless using RLHF.
- By following an iterative feedback approach that performs RLHF on a weekly basis with fresh data, ==authors find that they can train an LLM to be both helpful and harmless *without compromising performance on any benchmarks*==
	- Interestingly, the human feedback dataset that is curated in this paper is freely available online -- it's the ***==HH-RLHF Dataset==!***

Collecting Data:
- Feedback data is collected on a model's helpfulness and harmlessness, as judged by human annotators, on a prompt given to the model.
- Interestingly, we see that authors allow human annotators to interpret these terms loosely -- there are no detailed, itemized requirements written to further explain the meaning of helpful or harmless, allowing a large and diverse preference dataset to be collected.
	- ((This makes me think: Who are the annotators? Are they physicist PhDs from Anthropic?))

- **Helpful** data is collected by asking humans to solicit help from the model with a text-based task
- **Harmless** data is collected by asking humans to adversarially probe a model to get help with a harmful goal or use toxic language.

In all cases, *two model responses* are generated for each prompt, and the human annotator identifies the preferable response and a strength of preference based upon which response is more helpful or more harmless.

Training Setup
- The LLMs used have between 13M and 52B parameters.
- Experiments are performed with criteria-specific preference models, as well as preference models that are trained over a mixture of helpful and harmless data.
- Authors indicate that training preference models over a *mixture* of data that captures both alignment criteria (helpful, harmless) can achieve good scores in each metric. (though these two attributes are often in opposition, when considered singularly)
	- It seems that *large preference models are better able to simultaneously capture BOTH preferences!*

Despite collecting the strength of each preference score, preference models are trained to just assign a better score to the preferable output via a *==ranking loss==*
- The LLM is then optimized using the preference model + [[Proximal Policy Optimization|PPO]].

Does alignment degrade model quality?
- There has been a lot of of talk about the [[Alignment Tax]] -- whether the alignment procedure degrades the overall accuracy of the underlying LLM.
- The question is deeply related to the tension between helpfulness and harmlessness -- avoiding harmful output may cause the model to be less helpful on certain problems.

> "A question that's often raised about alignment training is whether it will compromise AI capabilities. We find that when RLHF is applied to large language models, the answer seems to be an almost-categorical no."

==We see that finetuning with RLHF does not necessarily deteriorate performance across more generic natural language benchmarks==. We see that *smaller models* may see a slight deterioration, but aligned models still perform quite well across other benchmarks... We learn that alignment doesn't always come at the cost of deteriorated performance on a broader set of tasks.

![[Pasted image 20240407231059.png]]

Takeaways and Analysis
- Although RLHF is not the focus of this overview, it's worthwhile to understand some of the major takeaways, as they provide insight into the properties of RLHF and how it can be automated with AI (i.e., RLAIF).

We learn a few useful things from this analysis:
- Smaller LLMs have an ==*alignment tax*== -- their performance deteriorates on other benchmarks after alignment with RLHF
- Larger models (13B, 52B) have an ==*alignment bonus*== -- their performance improves slightly!
- Alignment with RLHF is compatible with specialized language models. ==We can RLHF to models finetuned on code and it actually *improves* their coding abilities.==
- Larger preference models are better for making alignment more robust
- The iterative application of RLHF is effective:
	1. We collect new data
	2. We finetune the LLM to RLHF
	3. Redeploy this model to human annotators to collect more preference data on a weekly cadence.

### Constitutional AI: Harmlessness from AI Feedback \[1\]
- Link: https://arxiv.org/abs/2212.08073
- Research presented is quite interesting -- RLHF is a powerful tool for aligning language models based on human feedback, but it's difficult to scale up.
- ==Aligning a language model with RLHF requires a lot of human preferences labels, usually 10x more labels compared to a technique-like SFT.==
	- [[LLaMA 2]] uses 100,000 data points for SFT, but over 1,000,000 annotated examples are curated for RLHF!

With this in mind, we might wonder: Is there any way to automate creation of human preference labels for RLHF?
- ==Arguably, training of a reward model is already a form of automation!==
- This reward model is trained over human preference data, then used to generate preference labels during the reinforcement learning phase of RLHF.
- Authors set out with a goal of training a model that is helpful and harmless, and their approach, called [[Constitutional AI]] (CAI) leverages AI-provided feedback for collecting harmful preference data instead of humans.
	- We completely remove human feedback for identifying harmful outputs in an attempt to make obtaining preferences or feedback for alignment with RLHF both more scalable and explainable.

Writing the LLM constitution:
- 16 text-based principles are written (the constitution) which are then leveraged (with a few exemplars for few-shot learning) to automate the collection of preference data for harmfulness.
![[Pasted image 20240408001351.png]]
Above:
- CAI uses both SFT and RL for language model alignment
- Starting from an LLM that is purely helpful (i.e. doesn't have ability to avoid harmful output), we generate responses (which may be harmful) to a set of prompts, and then repeat the following steps:
	1. ==Randomly sample a single principle from the constitution==
	2. ==Ask the model to critique its response based on this principle==
	3. ==Ask the model to revise its response in light of this critique==
((Generate a response. Critique your response. Improve your response.))

![[Pasted image 20240408001753.png]]

After this constitution-based refinement process has been repeated multiple times for each prompt and response, we can then finetune the LLM (using SFT) over the set of resulting responses to make its output much less harmful.
- The purpose of this initial SFT is to get the model "on distribution," meaning that the model already performs relatively well and requires less exploration or training during the second phase of alignment.

After SFT has been performed, the LLM is further finetuned with Reinforcement Learning.
- This is identical to RLHF, but we replace human preferences for harmless (but not helpfulness; human annotations are still used for this criteria) with feedback provided by a generic LLM (This is RLAIF).
- For each prompt in a dataset of harmful prompts, the underlying LLM, which has already undergone SFT, is used to generate two responses. Then we generate a preference score using a generic language model (i.e. not the one that's undergone SFT) and the prompt template below

![[Pasted image 20240408001800.png]]


Again, we randomly sample a single principle from the constitution for each preference label that is created. All harmlessness preference labels are derived using a generic LLM via this multiple choice format.
- We create a dataset of soft preference labels by taking and normalizing the log probabilities of each potential response...

From here, we can perform a procedure nearly identical to RLHF, but with human-provided harmlessness data replaced by preference labels from an LLM.

Better prompt engineering
- To improve the automated feedback provided via the approach described above, authors test some more advanced prompting techniques.
1. First, utilizing few-shot examples within the critique and revision prompts used to generate eaxmples for supervised learning improves the quality of revised examples.
2. [[Chain of Thought]] prompting is found to improve the quality of revised responses.
3. Following a [[Self-Consistency]] approach that generates five responses via CoT prompting and averages the resulting preference labels yields a final, small performance boost.

We see that RLHF can be partially automated using AI-provided feedback with minimal performance degradation.
Using AI-generated labels for harmlessness can still yield improvements in the underlying LLM's harmlessness!


# RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback \[2\]
Link: https://arxiv.org/abs/2309.00267 (Sep 2023, Deepmind?)
- This technique is identical to RLHF, but it automates the creation of human preference labels by using an off-the-shelf LLM!
- RLAIF is explored specifically for text summarization tasks, and we see that the technique yields similar results when compared to RLHF, indicating that the alignment process can be automated via feedback provided by even a generic language model!
































