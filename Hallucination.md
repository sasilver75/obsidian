Refers to instances where the language model generates information that's not grounded in data or reality. When the output is fabricated, incorrect, or unsupported by provided data. We call them hallucinations because the model so confidently (as always) asserts them.

Hallucinations can be mitigated by using higher-quality datasets (diverse, accurate, minimally biased, domain-specific), aligning into the model (eg by RLHF) the ability to *refuse* to answer questions that it doesn't know, or by giving the model access to external knowledge sources (RAG).


OpenAI in a blog post talked about: (Ref: Ep 11 Zeta Alpha)
- Open-domain hallucination
- Closed-domain hallucination

Techniques to combat/ameliorate hallucination include ([from here](https://youtu.be/bCyWCz4NbN4?si=NhoM4XdghlOFxuW_)):
- Build tools for people to detect model-generated output
	1. Train an LM specifically for detection
	2. Use the source LM itself to detect its generations "zero shot" (eg looking at the probability), or perturb the generation and see how the model's probability responds... Methods like DetectGPT
- Help models better know what they don't know
	1. Abstaining
		- Avoiding incorrect predictions may be more important than classifying everything -- it might be better to be conservative.
		- Determining whether an input is out-of-distribution 
		- Selective classification: Only classify test inputs that the model are likely to get correct.
- Improve model predictions under some distribution shifts
	- Data rebalancing (upweight/upsample under-represented datapoints)
	- Domain invariance (learn representations invariant to domain)
(As you go down the list, you need more powerful tools/the problem gets harder)


From [[LLaMA 3.1]] paper:
> We develop a knowledge probing technique that takes advantage of L3's in-context abilities:
> 	1. ==Extract a data snippet== from pretraining data
> 	2. ==Generate a factual question== about the snippet with L3
> 	3. ==Sample responses== (plural) from L3
> 	4. ==Score *correctness*== of generations using the original context and L3 as a judge
> 	5. ==Score *informativeness*== of the generations using L3 as a judge
> 	6. ==Generate a refusal== for responses which are consistently informative and *incorrect* across generations (these are ones that seem to be deceptive to users) using L3.