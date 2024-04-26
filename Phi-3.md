April 22, 2024
Paper: [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)

A new Phi model family, basically scaling up [[Phi-2]]. 

Model family:
- Phi-3-mini (3.8B, 3.3T tokens)
	- Benchmarks as good as Mixtral, 3.5 Turbo, LLaMA3
	- Very similar to LLaMA 2 structure; same tokenizer as LLaMA and vocab size of ~32k
	- Can run quantized on an iPhone; Open weights, MIT license
	- Did chat fine-tuning (SFT, DPO); it's not just a base model
	- Uses [[LongRoPE]] to extend context
- Phi-3-small (7B, 4.8T tokens), still training at paper release time
	- In this case, stepped slightly away from LLaMA-style architecture, using [[Tiktoken]] tokenizer and a vocabulary size of ~100k.
	- [[Grouped Query Attention]]
- Phi-3-medium (14B, 4.8T tokens), still training at paper release time
	- 14B parameters

Training vibes
- Dataset is all you need
- Deviate from standard Chinchilla laws (over-trained regime, like LLaMA)
- Mix of heavily-filtered synthetic data
- 2 Phase training
	- Phase 1: Web data, learn "general knowledge and language"
	- Phase 2: Heavily filtered with web data + synthetic = learn "logical reasoning, niche skills"
- Introduce "Data Optimal Regime"
	- Quality of data for a given scale
	- How can we get the best quality data for a certain scale? 
	- Seems more efficient than LLaMA wit h their 2-tier approach

Does bad on knowledge stuff (eg Trivia QA), but if you hook it up to a RAG system, that would be great. It can do reasonable amounts of reasoning, though, so might be great on a mobile device.

Abstract
> We introduce phi-3-mini, a 3.8 billion parameter language model trained on 3.3 trillion tokens, whose overall performance, as measured by both academic benchmarks and internal testing, rivals that of models such as Mixtral 8x7B and GPT-3.5 (e.g., phi-3-mini achieves 69% on MMLU and 8.38 on MT-bench), despite being small enough to be deployed on a phone. The innovation lies entirely in our dataset for training, a scaled-up version of the one used for phi-2, composed of heavily filtered web data and synthetic data. The model is also further aligned for robustness, safety, and chat format. We also provide some initial parameter-scaling results with a 7B and 14B models trained for 4.8T tokens, called phi-3-small and phi-3-medium, both significantly more capable than phi-3-mini (e.g., respectively 75% and 78% on MMLU, and 8.7 and 8.9 on MT-bench).




![[Pasted image 20240424120925.png]]

![[Pasted image 20240424123058.png]]
