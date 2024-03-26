#article 
Link: https://newsletter.ruder.io/p/nlp-research-in-the-era-of-llms

-------

NLP research has undergone a paradigm shift over the last few years with LLMs -- most benchmarks are held by LLMS that are expensive to fine-tune, and too expensive to pretrain outside of a few industry labs.

In the past, a barrier to doing impactful research has often been a lack of awareness of fruitful research areas and compelling hypotheses to explore. ==In the era of expensive LLMs, what is left for academics, PhD students, and newcomers to the field?==

The state of research, thankfully, is NOT SO BLEAK!

-----
Aside:
- Based on:

![[Pasted image 20240325204449.png]]

----

# A cause for optimism
- Those who have been around for a long time often see old ideas reappearing in new guises, but with better-made costumes, of better materials -- research isn't linear, it's more like an ascending spiral!
- More recently, we've seen the success of scale with the advent of word embeddings in 2013 and the emergence of pre-trained LMs in 2018.

Main lesson:
- While massive compute often achieves breakthrough results, its *usage* is often insufficient! Over time, improved hardware, new techniques, and novel insights provide opportunities for dramatic compute reduction.


There are already examples that require a fraction of compute by using new methods and insights, demonstrating that this trend also holds in the era of LLMs:
- [[FlashAttention]]: Provides dramatic speedups over standard attention through clever hardware optimization.
- [[Parameter-Efficient Fine-Tuning|PEFT]] techniques like [[Low-Rank Adaptation|LoRA]] and [[Quantized Low-Rank Adaptation|QLoRA]] enable finetuning LLMs on a single GPU.
- [[Phi-2]] *seems to* match/outperform models significantly (~25x?) larger

In the near term, ==the largest models using the most compute will continue to be the most capable, but there remain a lot of room for innovation by focusing on strong smaller models, and on areas where compute requirements will inexorably be eroded by research progress.==


Reminder: ==Researchers do not need to assemble full-fledged, expensive systems in order to have impact!==
> "Aerospace engineering students are not expected to engineer an entire new airplane during their studies."
> - Chris Manning

Let's look at five important research areas requiring less compute.

# 1. Efficient Methods
- When we think about efficiency, we often think about making the *model* architecture more efficient (often-times focused specifically on the attention mechanism) -- but ==it's **actually useful to consider the entire LLM stack!***==

Important components ripe for improvements are:
1. ==Data Collection and Preprocessing== (Better filtering/data selection)
2. ==Model Input== (Faster, more-informed tokenization, better word representations w/ *character-level* modeling)
3. ==Model Architecture== (Better scaling to long-range sequences)
4. ==Training== (More efficient methods to train small-scale LLMs via better distillation, LR schedules and restarts, (partial) model compression, model surgery, etc.)
5. ==Downstream task adaptation== (Improved PEFT, automatic prompt and [[Chain of Thought|CoT]] design, improved RLHF)
6. ==Inference== (Early predictions, prompt compression, human-in-the-loop interactions)
7. ==Data annotation==: Model-in-the-loop annotation, automatic arbitration and consolidation of annotations
8. ==Evaluation==: Efficient automatic metrics, efficient/more reasonable benchmarks.

Given the wide range of LLM applications, it's important to consider the HUMAN part in efficiency
- How can we make the stages where human and LLM data intersect (preferences, annotation, interacting with users) more efficient and reliable?

Sparsity and lower-rank approximations are two principles that have been applied in a wide range of efficient methods, and are good sources of inspiration.

A good measure of whether an efficient method works is whether it reduces the coefficient/lowers the slope of the corresponding scaling law
![[Pasted image 20240325210654.png|300]]



# 2. Small-scale Problems
- While it's generally prohibitive to apply a new method directly to larger models, using it on smaller, representative models can serve as a useful prototype and proof of concept!
	- e.g. the recently proposed [[Direct Preference Optimization|DPO]] method used a smalls-scale experimental setting (GPT-2-large fine-tuned on IMDB reviews). 
- Another setting where a focus on a small scale is on analysis/model understanding -- when do capabilities seem to spontaneously emerge?
- Small and controlled settings that allow probing of specific hypotheses will be increasingly important to understand how LLMs learn and acquire capabilities.

LLM mechanisms that remain poorly understood include:
- In-context learning (highly-skewed distribution of language data are important?)
- Chain of Thought prompting (Local structure in the training data is important, but why?)
- Cross-lingual generalization
- Other types of emerging abilities (Mirage paper @ NIPS)


# 3. Data Constrained Settings
- The largest LLMs are trained on trillion s of tokens -- but in many domains (Science, Education, Law, Medicine), there is very little high-quality data easily accessible online! LLMs thus must be combined with domain-specific strategies to achieve the biggest impact.
- Multilinguality is another area where data is notoriously limited -- Many languages and dialects are more commonly spoken than written, which makes multi-modal models important to serve such languages.
- As we reach the limits of data available online, even high-resource languages will face data constraints -- new research will need to engage with these constraints rather than assume an infinite scaling setting.


# 4. Evaluation
- In 2021, a common sentiment was that NLP models had outpaced the benchmarks to test them.

> "*Benchmarks shape a field, for better or worse -- Good benchmarks are in alignment with real applications, but bad benchmarks are not, forcing engineers to choose between making changes that help end-users, or making changes that only help with marketing.*"
> - David Patterson, System Benchmarking (2020)







