August 6, 2023
UC Santa Barbara (*Pan et al.*)
[Automatically Correcting Large Language Models - Surveying the landscape of diverse self-correction techniques](https://arxiv.org/abs/2308.03188)
#zotero 
Takeaway: ...

---

## Introduction
- (Description of the usual problems of LLMs)
- To minimize the need for human intervention *self-correcting LLMs using automated feedback* , where the model learns from automatically-generated external feedback signals that could come from:
	1. ==The LLM itself== acting as a feedback model ((*intrinsic* self-correction, according to GDM))
	2. ==A separately-trained feedback model==
	3. Readily-available ==external tools==
	4. ==External knowledge sources== such as Wikipedia or the internet
	5. ==External evaluation metrics==

## Taxonomy for Correcting LLMs with Automated Feedback
- ==Language Model== (*Patient*): Maps an input $x$ to an output $\hat{y}$, which can be different depending on NLP task.
- ==Critic Model== (*Doctor* and *Diagnosis*): Learns to generate feedback from $(x, \hat{y}) \rightarrow c$ . This feedback is in some specific format, e.g. a scalar value, natural language, or binary feedback.
- ==Refine Model== (*Treatment*): Learns to repair an output based on feedback $(x,\hat{y}, c) \rightarrow y_{new}$ . Besides repairing the *output*, some refine models directly repair the *language model* itself through fine-tuning or reinforcement learning.
- What gets corrected?
	- ==Hallucinations== can be corrected by cross-referencing model output with credible knowledge sources.
	- ==Unfaithful Reasoning== (where the conclusion doesn't follow from the previously-generated reasoning chain) can be corrected by using feedback from external tools or models for guiding/verifying the reasoning process.
	- ==Toxic, Biased, and Harmful Contents== can be rectified (by RLHF, but also) collecting automated feedback to identify and correct potentially harmful outputs (See *CRITIC* paper).
	- ==Flawed Code== (or Math) can addressed by using automated feedback from code execution environments (or math solvers).
- What's the source of feedback?
	- Can be divided into two categories:
		1. Human feedback
		2. Automated feedback
			1. Self-feedback (feedback originates from the LM itself)
				- Evaluate generated outputs through prompting and subsequently use this feedback to refine the results (can be done iteratively).
					- (([[DeepMind|GDM]] and [[Subbarao Kambhampati]] are critical of this))
			2. External feedback (external models, tools, knowledge sources)
				- Originates from other trained models, external tools, external knowledge sources, and external evaluation metrics.
- What's the format of the feedback?
	- Most typically in the format of a scalar value signal or natural language
	1. Scalar Value Feedback
		- A single score that can be easily integrated into the training/decoding process of LLMs; The Self-Verification paper ranks candidate outputs to find the optimal based on a real-value feedback score assigned by a critic model.
	2. Natural Language Feedback
		- 


## Training-Time Correction


## Generation-Time Correction


## Post-hoc Correction


## Applications


## Research Gaps and Future Directions

Abstract
> Large language models (LLMs) have demonstrated remarkable performance across a wide array of NLP tasks. However, their efficacy is undermined by undesired and inconsistent behaviors, including hallucination, unfaithful reasoning, and toxic content. A promising approach to rectify these flaws is ==self-correction==, where the ==LLM itself is prompted or guided to fix problems in its own output==. Techniques leveraging ==automated feedback -- either produced by the LLM itself or some external system== -- are of particular interest as they are a promising way to make LLM-based solutions more practical and deployable with minimal human feedback. ==This paper presents a comprehensive review of this emerging class of techniques==. We analyze and taxonomize a wide array of recent work utilizing these strategies, including ==training-time==, ==generation-time==, and ==post-hoc correction==. We also summarize the major applications of this strategy and conclude by discussing future directions and challenges.


# Paper Figures
![[Pasted image 20240806235424.png]]
