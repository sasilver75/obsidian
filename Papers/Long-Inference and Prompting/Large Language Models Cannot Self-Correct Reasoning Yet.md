October 3, 2023
[[DeepMind|Google DeepMind]], UIUC (Huang et al.)
[Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/pdf/2310.01798v2)
#zotero 
Takeaway: Authors contest that LLMs contest that yet self-correct reasoning in an intrinsic self-correction setting, where the LM isn't given external feedback from an oracle (model) or tools/knowledge base, and is just asked to critique and then self-correct their answer. Authors say that papers that would have you believe otherwise are often misleading. Authors claim that self-correction can be useful for stylistic revisions, but not for reasoning-related ones. Oracle use resulted in improved generations. Intrinsic self-correction resulted in performance *decreases*. Multi-agent debate for the purposes of self-correction is no better than [[Self-Consistency]].

Note: The ideas in this paper is similar to the ideas often espoused by [[Subbarao Kambhampati]] and his research team.


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
### Self Correction with Oracle Feedback
- Benchmarks: Focusing on [[GSM8K]] (grade-school math word problems; a previously-reported 7% improvement post self-correction), [[CommonSenseQA]] (MC questions for commonsense reasoning, with a previously-claimed 15% increase through self-correction), and [[HotpotQA]] (Multi-hop QA dataset, "significant performance improvement" priorly reported with self-correction)
- Prompts: We apply a three-step prompting strategy for self-correction (eg like [[Reflexion]]):
	1. Prompt the model to perform an initial generation
	2. Prompt the model to review its previous generation and produce feedback.
	3. Prompt the model to answer the original question again with feedback
For experiments, we mostly adhere to the prompts from the source papers (for the papers claiming the improvements on these benchmarks), using [[GPT-3.5]] Turbo (but also on [[GPT-4]]).
- We find similar ==significant performance improvements== for each benchmark, like the authors did. Recall that this is *with an oracle.* In a realistic scenario, the correct answer will be unknown to us -- in our scenario, we're stopping when the model reaches the correct answer (which is unrealistic that we're "telling" the model "Yep, you got it right!")
- ==Authors try a similar experiment where instead of having an oracle give feedback, they corrective action is derived from *random guessing* from the *remaining* options... And it performs ~as well or even better than self-correct using an oracle!== (table 2)

### Intrinsic Self-Correction
- In this scenario, we remove the use of labels to determine when to stop, and evaluate the performance with (up to) two rounds of self-correction.
- Results show that after self-correction (whether one round or two rounds), ==performance drops across all benchmarks== (Figure 1), with the second round of self-correction not making an obvious positive or negative impact on the damage caused by the first round.
- For GSM8K, 74.7% of the time, the model retains its initial answer, but among remaining benchmarks, the model is more likely to modify a correct answer to an incorrect one than to revise an incorrect answer to a correct one; the self-correct prompt might bias the model to choose other options.
	- If the model is well-aligned and paired with a good initial prompt, the initial response should already be optimal given the conditions of the prompt and the specific decoding algorithm; feedback can be viewed as adding an additional prompt, potentially skewing the model towards generating a response that's tailored to this combined input.
- If the self-correction prompt we tested is suboptimal, could other prompts lead to an improved performance? Possibly - but this no longer aligns with the *intrinsic self-correction setting* discussed in this paper; such a search essentially leverages feedback from humans or training examples.
	- To be clear, our focus is not on "Is there a self-correction prompt that can bolster performance on specific benchmarks?" -- Such a query might not be particularly meaningful. Instead, we want to tackle "Are LMs able to self-correct reasoning *based solely on their inherent capabilities*?"

### Multi-Agent Debate and Self-Consistency
- Another approach in the literature involves allowing the models to critique and debate through multiple model calls.
- Du et al (2023) implement a multi-agent debate method, leveraging multiple instances of a single ChatGPT model and demonstrate significant improvements on reasoning tasks.
- The results indicate that both multi-agent debate and [[Self-Consistency]] achieve significant improvements over standard prompting... 
	- Performance of multi-agent is only slightly better than that of self-consistency with the same number of agents. ((? Unclear))
	- For self-consistency with an equivalent number of responses, ==multi-agent debate significantly underperforms simple self-consistency using majority voting.==
- ==Authors say that multi-agent debate is really a method to achieve 'consistency', across multiple generations, with the distinction (between it and self-consistency) lying in the voting mechanism, whether voting is model-driven or purely based on counts.==

## Self-Correction as Post-Hoc Prompting
- Some papers ([[Self-Refine]], [[Constitutional AI|CAI]]) note impressive results from self-correction -- it's important to pinpoint the underlying causes of this.
- Self-correction can be viewed as a type of ==post-hoc prompting==, where the prompting is conducted *on top of the responses from LLMs.* We refer to the process of improving such prompts as ==post-hoc prompt engineering.==
- Scenarios in which self-correction enhances model responses occur when it can provide valuable instruction or feedback that pre-hoc prompting cannot.
	- ==When the goal is to make the response safer==, it might be challenging to instruct a model to generate completely risk-free responses in its first attempt, using only pre-hoc prompting; in this case, self-correction can serve as a means to enhance the safety of responses through post-hoc examination.
		- Even when significant performance improvement post self-correction is observed,  careful consideration of prompt design is essential!
	- ==For reasoning tasks, this might not be the case==; "Review your previous answer and find problems with your answer" does not necessarily provide tangible benefits for reasoning.
	- Even when performance improvement post self-correction is observed, consideration of prompt design is essential; If the response needs to meet criteria that could have been easily specified in the initial instruction (eg don't contain negative words), instead of feeding these requirements as feedback in the post-hoc prompt, a more cost-effective alternative strategy is to embed these directly into the pre-hoc prompt.
	- ==It is meaningless to employ a well-prompted post-hoc prompt to guide the model in "self-correcting" a response generated through a poorly-constructed pre-hoc prompt!==

## Discussion
- Self correction may still be beneficial for aligning responses with certain preferences.
	- Self-correction isn't useless; it can be effectively employed to make models align with specific preferences, such as altering the style of responses or enhancing their safety. 
	- ==It's good for style, not correctness/reasoning== ((This is echoed by Subbarao Kambhampati))
- Leveraging external feedback for correction
	- In this paper, we've focused on intrinsic self-correction; but when we leverage external feedback for correction, the narrative changes!
		- Calculators, search engines, code executors can help LLMs improve their generation to better-solve reasoning tasks!
	- The [[Shepherd]] team trains a verifier/critique model on a high-quality dataset to verify and refine LLM outputs.
- Employing [[Self-Consistency]] as a method of self-verification
	- ==We observed above that the oracle setting yields higher accuracy, suggesting that in an LLM's search space, a correct answer *might exist*... if we can leverage a robust verification process to guide LLMs toward the right direction (or away from bad ones) outcomes can be enhanced.==
	- This can either be done via external feedback of self-consistency; an example of this is [[Tree of Thoughts]], where LLM reasoning is enhanced through step-wise verification paired with self-consistency. 
- Pre-hoc vs Post-hoc prompting
	- It's preferable to place greater emphasis on pre-hoc prompt engineering than on costlier post-hoc prompt engineering, though in cases where you have to incorporate external feedback, you'll need post-hoc prompting.
- ==Guidelines for comparison==
	- When comparing self-correction methods, it's important to report the inference cost/number of calls/tokens.
	- It's advisable to include [[Self-Consistency]] with the same number of calls/responses as a baseline.
	- ==Avoid using an ill-designed pre-hoc prompt with a carefully-designed post-hoc prompt for improvement.==


## Conclusion, Limitations, and Broader Impact
- ==Our research shows that LLMs are not yet capable of intrinsically self-correcting their reasoning.==
- Several related works have already presented findings consistent with ours (Gou 2023 "Critic", Zhou 2023)
- Authors urge that people take a more wary view on unwarranted optimism/fear regarding autonomous evolution of LMs through (intrinsic) self-improvement.



Abstract
> Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, ==self-correction==, has been proposed as a remedy to these issues. Building upon this premise, ==this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations==. Central to our investigation is the notion of ***==intrinsic self-correction==***, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.


# Paper Figures
![[Pasted image 20240731113153.png]]
See mild improvement using self-correction *with an oracle*. We observe significant performance improvements, consistent with the findings of other papers.

![[Pasted image 20240731113505.png|300]]
Authors try a similar experiment where instead of having an oracle give feedback, they corrective action is derived from *random guessing* from the *remaining* options... And it performs ~as well or even better than self-correct using an oracle!

![[Pasted image 20240731113657.png|500]]
In this intrinsic self-correction scenario, see that adding in a self-correction round(s) results in *worse* performance across all benchmarks, with CommonSenseQA being affected the most.
- False answer options in CommonSenseQA often appear somewhat relevant to the question, and authors think that the self-correction step results in the model being incorrectly biased to choose another option, leading to a high correct -> incorrect ratio.

![[Pasted image 20240731114552.png|400]]
Examples of a successful self-correction (wrong -> right) and an unsuccessful self-correction (right -> wrong).

![[Pasted image 20240731140055.png|500]]
For self-consistency with an equivalent number of responses, ==multi-agent debate significantly underperforms simple self-consistency using majority voting.==

![[Pasted image 20240731140620.png|400]]

![[Pasted image 20240731145459.png]]
Example of a GOOD incorrect -> correct self-critique on a math problem

![[Pasted image 20240731145526.png]]
Example of a BAD correct -> incorrect self-correction on GSM8K

![[Pasted image 20240731150142.png]]
Example where self-correction results in no change

![[Pasted image 20240731150158.png]]
A BAD example in CommonSense QA (correct -> incorrect)


