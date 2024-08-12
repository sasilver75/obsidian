https://www.interconnects.ai/p/frontier-model-post-training
From [[Nathan Lambert]]

----

When people look for the "Default RLHF Recipe", they find [InstructGPT](https://arxiv.org/abs/2203.02155), [WebGPT](https://arxiv.org/abs/2112.09332), [Sparrow](https://deepmind.google/discover/blog/building-safer-dialogue-agents/), [Summarizing from Human Feedback](https://arxiv.org/abs/2009.01325), and [Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862).

Nathan has talked about the =="10k high quality instructions and 100k preferences"== idea so many times.

These were seminal works but are becoming relatively out of date with how RLHF is done today -- directionally, they're still relevant, but the details aren't accurate representations of reality.

The following papers:
- [[LLaMA 3.1]]
- [[Nemotron-4]]
- [[Apple Intelligence Foundation Language Models|Apple Foundation Model]]
- [[Gemma 2]]
have made it clear that a new default recipe exists for doing high-quality RLHF.

This recipe requires some assumptions:
- ==Synthetic data can be of higher quality than humans, especially for demonstrations on challenging tasks.==
- ==RLHF can scale far further than instruction tuning.==
- It takes *==multiple rounds of training and generation==* to reach your best model.
- ==Data filtering== is the ***most important part of training***.

----

## Human Preference Data
- The focus of original RLHF pipeline was on human data, coming in two forms:
	1. Human data for ***==instruction-tuning==*** on specialized tasks
	2. Human ***==preference==*** data on model completions
- These datasets were COSTLY and GUARDED, with few examples available in the open source.
- Human preference data was tied to improving a specific model, and, to the extent that data *could* be open, ==we're not sure that we could transfer one model's online preference data to another later== -- we tried this at HF and failed on a small paid data contract.
- From the [[LLaMA 2]] numbers, it's likely that Meta spent $10-$20 million or more on preference data.
	- [[Nemotron-4]] used a large amount of synthetic data to replace this, but this model is largely not seen as a comparably strong fine-tune.

==A challenge in this area for the open community is to figure out how much of this human intervention can be completed with [[LLM-as-a-Judge]] or [[Reward Model]]ing techniques.==


## Scaling RLHF
- On a Latent Space podcast, Thomas Scialom (alignment lead on LLaMA 3) said:
> "RLHF is so much more scalable. It costs less, it's easier, and that leads in general to just better performance."
- He continued to say that he would spend *100% of his alignment data budget on preferences*, rather than spend more time on instructions!

==Far too many in the open source alignment space are focused on scaling instruction fine-tuning.==
- This is approachable, works for many tasks, and is simple to use with synthetic completions, but it's clear that *industry* just sees IFT as a starting point that you then scale from.
- The SFT data is focused around *specific domains* (eg code, math), and then they use RLHF to scale from there.

==RLHF is an iterative process==
- [[LLaMA 2]] and [[Nemotron-4]] papers detail about 5 training rounds, and we don't know what the training ceiling is on this.
- [[LLaMA 3.1]] was 6 rounds of preference training, with multiple instruction tuning rounds coming before.

Multiple rounds of human preference data training is likely to mostly be a practicality, because:
1. Data is delivered to the lab from the annotation company in rounds.
2. Small training runs de-risk the end product -- it's nice to see that you're on track over time, rather than waiting for all data to come in before pressing "go."


![[Pasted image 20240811143323.png]]
Above: Figure from the [[LLaMA 2]] paper where they had 5 RLHF rounds of [[Rejection Sampling]] and [[Proximal Policy Optimization|PPO]].

![[Pasted image 20240811143440.png]]
Above: [[Nemotron-4]] did two SFT rounds and 4 alignment rounds. Their [[Reward-Aware Preference Optimization|RPO]] is just NVIDIA's reward-model-weighted [[Direct Preference Optimization|DPO]] optimizer.

Iterative RLHF approaches go all the way back to Anthropic's [[Constitutional AI|CAI]] paper, but are ==largely not reproduced in the open-source community.==

Academics are focusing on "[online DPO trainers](https://www.interconnects.ai/p/how-rlhf-works-2?open=false#%C2%A7some-next-steps-for-rlhf-research)" these days, which is directionally similar, but comes with much less focus on data between rounds.

The algorithm choices of the various teams should not be fixated on -- ==DPO scales more easily, but PPO-inspired methods (online RL) have a higher ceiling==.

LLAMA 3 has a simple post-training loop:
- Rejection Sampling, SFT, and DPO
- > "This not only performed best empirically, but it also enabled reproducibility, and our team could asynchronously explore many different workstreams (coding, math), funneling data in to the same, simple loop."


## Synthetic Data
- A large proportion of the new RLHF loop is only made possible by synthetic instruction data surpassing the capabilities of humans on most tasks.
- Meta made it clear that they used the 405B model to improve the post-training quality of the smaller [[LLaMA 3.1]] models.
	- Google does this distillation similarly to [[Gemeni Flash]].
- ==Rumors that OpenAI is training its next generation of models on 50T tokens of largely synthetic data.==
	- Last year, there were rumors that ==Anthropic had a pretraining-scale Constitutional AI corpus== -- these now seem more credible.

Claims of "[model collapse](https://www.nature.com/articles/d41586-024-02420-7)" are [overblown](https://x.com/RylanSchaeffer/status/1816881533795422404) -- it only leads to model collapse when deployed in a contrived setting that doesn't reflect reality.


## Data Quality is King
- The majority of the [[LLaMA 3.1]] report are about data curation details. Every subdomain of interested needs *extensive and specific curation instructions!*
- At AI2, we started prioritizing data quality far more for our post-training recipes, and could ==immediately feel the pace of progress shift.== 


## Putting it all together

![[Pasted image 20240811171948.png]]
Above: LLaMA's feedback loop around synthetic data and optimization

![[Pasted image 20240811172307.png]]
Above: Nemotrons

![[Pasted image 20240811172330.png]]



## Apple confirms the new normal
- Apple's recent launch of the beta version of Apple Intelligence was accompanied by the afore-linked technical report for its foundation models. 
- It doesn't go into as much detail as Nvidia and Meta, but covers enough to be clear...
	- Both human and synthetic data
	- We found data quality to be the key to model success and thus have conducted extensive data curation and filtering procedures.
They do the new normal, including:
- Human preference data and [[HelpSteer]] style grading of attributes for regularization.
- High-quality [[Reward Model]]s for filtering
- Replacement of human demonstration with model completions in some domains
- ==Multi-round RLHF==
- A very large suite of ==data curation techniques==, including ==prompt re-writing== and refining for expansion of costly datasets, filtering math and code answers with outcomes (correctness or execution), filtering with LLMs-as-a-Judge, and other new normal stuff.

Apple includes the fundamentals of their RL methods, including a different type of soft margin loss for the reward model, regularizing binary preferences with absolute scores, their iTeC rejection sampling algorithm that is very similar to Meta’s approach, and their leave-one-out Mirror Descent RL algorithm, MDLOO

Some notes on them, because I can’t help myself:

- The rejection sampling (iTeC) uses a large amount of models to generate completions.
    
- The rejection sampling (iTeC) uses just the best completion of the batch for each prompt, to make sure each prompt is in the instruction tuning step.
    
- Apple heavily misuses the technical term distillation in their report, as many of us now do, now that knowledge distillation is popularized again.
    
- The RL algorithm MDLOO algorithm [builds on interesting work from Cohere](https://arxiv.org/abs/2402.14740) — it seems promising.




