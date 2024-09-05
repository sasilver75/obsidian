May 2, 2024
[[KAIST]], LG AI Research, CMU, MIT, AI2, UIC
Paper: [Prometheus 2: An Open-Source LM specialized  in evaluating other LMs](https://arxiv.org/abs/2405.01535)
#zotero 
Takeaway: The next paper in a line of work trying to train open models to replace GPT-4 et al. for LLM-as-a-Judge. It's a very promising model, but weirdly Nate Lambert said that it didn't score especially well on RewardBench. It's notable because it can do both ==direct assessment== (single response scalar rating) and ==pairwise ranking== (which of two generations is better?).

---

Notes:
- In the abstract, the authors note three shortcomings of open-source LMs specializing in evaluation:
	1. Their scores don't highly correlate with human scores
	2. They lack the flexibility to do both ==direct assessment== *and* ==pairwise ranking==.
	3. They don't possess the ability to evaluate based on custom evaluation criteria, instead focusing on general attributes like helpfulness and harmlessness.
- Prometheus can do all of these things! It scores the highest correlation with humans and proprietary LM judges, among all of the open-source evaluator LMs.
	- It can perform both direct assessment *and* pairwise ranking, using a user-defined evaluation criteria.
- In a ==direct assessment==, LMs are prompted to output a scalar indicator of quality.
- In ==pairwise ranking==, LMs determine which of two outputs are preferred.
- Relying on proprietary LMs (which are the only ones for which this really works, having high agreement with humans) for evaluation poses significant challenges -- there's a lack of transparency, controllability, and affordability.
	- Unfortunately, most open LMs don't yet yield scoring decisions that correlate well enough with human judgements to replace them.
- To close the gap with proprietary LMs authors investigate *unifying* the two model-based evaluation paradigms (direct assessment and pairwise ranking) to train a robust, unified evaluator LM. They do this by *==merging==* weights of two evaluator LMs trained separately on direct assessment and pairwise ranking formats.
	- ==We observe that weight merging can yield an evaluator LM that not only *works* in both formats, but also *outperforms* evaluator LMs that are jointly trained, or only trained on a single format!==
- Authors develop the ==[[Preference Collection]]== fine-grained pairwise ranking feedback dataset that builds on [[Feedback Collection]] (which was a direct assessment feedback dataset).
- Authors use [[Mistral]]-7B and [[Mixtral]]-8x7B as base models, and merge the weights of 2 evaluator LMs separately finetuned (respectively) on Feedback Collection and Preference Collection to obtain our resulting [[Prometheus 2]] models (7B and 8x7B).
- On four direct assessment benchmarks, the Prometheus 2 models demonstrate the highest correlation with both human evaluators and proprietary evaluator LMs.
- Direct Assessment
	- Mapping an instruction *i* and response *r* into scalar value score *s*. For our scoring range, we use a 1-5 Likert scale scoring.
	- Prior works have shown that it's crucial to add a reference answer *a* as input to maximize correlation between LM and human. Prompting LMs to write verbal feedback before giving a score also improves correlation. Lastly, by using a scoring rubric evaluation criteria, users can ensure their models are flexible to specific needs, rather than generic qualities.
- Pairwise Ranking
	- Pairwise ranking is mapping an instruction and two pairs of responses into some ranking order. 
	- Similar to direct assessment, prior works have shown that integrating a reference answer and verbal feedback into the evaluation pipeline is crucial. We also add evaluation criteria as input to the evaluator LM.
	- We don't include a set of descriptions for each score; instead, only the description of the evaluation criteria itself.
- Weight Merging
	- Prior works demonstrated that weight merging can enhance performance across various domains, including language modeling, instruction-tuning, and aligning to user preferences.
	- By merging models trained on different assessment formats (here, direct assessment and pairwise ranking), we aim to obtain an evaluator LM that not only functions in both formats, but also shows as good evaluation performances as proprietary LMs.
	- We obtain a final evaluator LM with "linear merging"
		- $\theta_final = \alpha \times \theta_d + (1-\alpha) \times \theta_p$  
	- They tried some other techniques (task arithmetic merging, TIES merging, DARE merging) too.
- Preference Collection
	- Popular pairwise ranking datasets like [[Helpful and Harmless|HH]]-RLHF or [[UltraFeedback]] do not include an evaluation criteria and verbal feedback -- [[Preference Collection]] does!
	- To construct Preference Collection, we apply to modifications to [[Feedback Collection]] (direct assessment dataset).
		- Since Feedback Collection includes five responses for each instruction, each with a score in [1...5], we pair two out of the five responses, resulting in a total of ten combinations per instruction.
		- Using the existing scoring decisions for each response, we determine which is better and assign a new scoring decision for that pair.
		- To generate new feedback for each pair of responses, we prompt GPT-4 to identify the commonalities and differences of the two responses.
- Results
	- Direct Assessment: Prometheus-2 has high correlation with GPT-4, Claude3Opus (each correlating with eachother with Pearson correlations higher than .5 regardless of the reference evaluator and benchmark)
	- Pairwise Ranking: Prometheus2 models achieve the highest scores, showing that they could effectively simulate human judgements. 
- Discussions:
	- Is Weight training more effective compared to joint training?
		- We observe that evaluator LMs trained via joint training often show lower performance compared to single-format trained evaluator LMs, indicating *negative task transfer*.
		- LMs trained via weight merging show superior performance not only compared to jointly-trained evaluator LMs, but also single-format trained evaluator LMs, indicating *positive task transfer!* ==This is super interesting! Meaning there's negative transfer when jointly training, but positive transfer when weight merging!==
	- Is the effectiveness of weight training due to model ensembling?
		- It's unclear why weight merging works -- maybe it's due to the effect of ensembling multiple models? We do an experiment where we train multiple evaluator LMs on different random seeds and merge them -- in the majority of cases, this doesn't improve evaluation performance.
		- Interesting, merging two evaluator LMs trained on direct assessment *harms* performance on average -- same for pairwise ranking. But if you merge two evaluator LMs with one trained on each, performance improves. This suggests that positive task transfer occurs from weight merging models with different evaluation formats, not by ensembling multiple models.
	- To what extent does learning with direct assessment help pairwise ranking performance, and vice versa?
		- They varied the $\alpha$ value in their linear merging, and looked at performance on direct assessment; the found it was optimal when alpha was set to .5; Interestingly, for pairwise ranking, they found optimal performance at .3.


Abstract
> Proprietary LMs such as GPT-4 are often employed to assess the quality of responses from various LMs. However, concerns including transparency, controllability, and affordability strongly motivate the development of opensource LMs specialized in evaluations. On the other hand, ==existing open evaluator LMs exhibit critical shortcomings==: 1) they ==issue scores that significantly diverge from those assigned by humans==, and 2) they ==lack the flexibility to perform both direct assessment and pairwise ranking==, the two most prevalent forms of assessment. Additionally, they ==do not possess the ability to evaluate based on custom evaluation criteria==, focusing instead on general attributes like helpfulness and harmlessness. To address these issues, we introduce ==Prometheus 2==, a more powerful evaluator LM than it’s predecessor that ==closely mirrors human and GPT-4 judgements. Moreover, it is capable of processing both direct assessment and pair-wise ranking formats grouped with a user-defined evaluation criteria==. On four direct assessment benchmarks and four pairwise ranking benchmarks, PROMETHEUS 2scores the highest correlation and agreement with humans and proprietary LM judges among all tested open evaluator LMs. Our models, code, and data are all publicly available


# Paper Figures
![[Pasted image 20240516214857.png]]
Above: Showing the real gap between the "weak evaluator group" and the "strong evaluator group" (which includes frontier models, humans, and, apparently, Prometheus 2 🤪)

![[Pasted image 20240516222704.png]]
Above: See this diagram comparing/contrasting direct assessment and pairwise ranking.
- Pairwise Ranking: Given two responses and an evaluation criteria, produce (optionally) feedback, then a scoring decision.
- Direct Assessment: Given a single response and evaluation criteria, produce (optionally) feedback, then a scoring decision.

![[Pasted image 20240516224400.png]]

![[Pasted image 20240516230931.png]]
Above: Seems weight merging is pretty goated in most situations. I wonder why the disparity was so large in some of the direct assessment benchmarks? Does this mean that jointly training *harms* direct assessment capability (but doesn't really hurt pairwise ranking)?

![[Pasted image 20240516233044.png]]
Above: From the linear merging strategy, playing with the $\alpha$ and seeing how it affects performance on direct assessment. See that an optimal mix seems to be 50/50!