October 3, 2023
[[DeepMind|Google DeepMind]], UIUC (Huang et al.)
[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/pdf/2310.01798v2)
#zotero 
Takeway: ...

---

## Introduction
- Amid concerns about LLM reasoning capability, the concept of *==Self-Correction==* has emerged as as promising idea, ==where an LLM refines its responses based on feedback to their previous outputs==.
	- Question arises: ==If an LLM answers the ability to self-correct why doesn't it simply offer the correct answer in the first place?== This paper delves into this paradox!
- We define the concept of ***==intrinsic self-correction,==*** in which the model tries to rectify its initial responses solely on inherent capabilities, without the crutch of external feedback.
	- High-quality external feedback is often unavailable, in many renewal-world settings.
	- ==Contrary to the optimism surrounding self-correction, our findings indicate that LLMs struggle to amend their prior responses in this setting!== In most instances, the performance post self-correction even *==deteriorates==!*
		- This observation is in contrast to prior research like [[Reflexion]], [[Self-Refine]], etc.
		- GDM authors note that the improvements in these studies result from using *oracles* to guide the self-correction process, and the improvements vanish when oracle labels are not available.
	- We also investigate the potential of ==*multi-agent debate*== as a means of improving reasoning, in which multiple instances of ((the same?)) LLM critique eachothers' responses -- but our results reveal that its efficacy is no better than [[Self-Consistency]] when considering an equivalent number of responses, highlighting the limitations of such an approach.
- While self-correction has limitations in enhancing reasoning, it does show promising results in other tasks like altering the *style* or *appropriateness* of responses.

## Background and Related Work
- The discourse on self-correction pivots around whether these LMs can recognize the appropriateness or accuracy of their outputs, and, if needed, provide refined answers.
	- In an ideal self-correction scenario (eg in math), a model might recognize the error in one of its calculation steps, revisit the problem, correct the error, and consequently produce a more accurate solution.
	- Various terms like "self-refine," "self-critique," "self-improve" and others have emerged, all with their own specific context (leading to some ambiguity).
	- A pivotal distinction lies in the *source of feedback*:
		1. Is it purely internal, originating from the LLM and relying on its inherent knowledge?
		2. Does it draw from external inputs? From humans, other models, external tools or knowledge sources?
			- Models: [[Shepherd]], Refiner (2023)
			- External tools and knowledge sources: Critic (2023), Teaching LLMs to Self-Debug (2023), Rarr (2023)
		- This paper is going to focus on (1), whether LLMs have the inherent ability to correct their responses, without external or human feedback, a setting which we call ==**intrinsic self-correction***==.


## Can Large Language Models Self-Correct Reasoning?
Self Correction with Oracle Feedback
- Benchmarks: Focusing on [[GSM8K]] (grade-school math word problems; a previously-reported 7% improvement post self-correction), [[CommonSenseQA]] (MC questions for commonsense reasoning, with a previously-claimed 15% increase through self-correction), and [[HotpotQA]] (Multi-hop QA dataset, "significant performance improvement" priorly reported with self-correction)
- Prompts: We apply a three-step prompting strategy for self-correction (eg like [[Reflexion]]):
	1. Prompt the model to perform an initial generation
	2. Prompt the model to review its previous generation and produce feedback.
	3. Prompt the model to answer the original question again with feedback

## Self-Correction as Post-Hoc Prompting


## Discussion


## Conclusion, Limitations, and Broader Impact



Abstract
> Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, ==self-correction==, has been proposed as a remedy to these issues. Building upon this premise, ==this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations==. Central to our investigation is the notion of ***==intrinsic self-correction==***, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.
