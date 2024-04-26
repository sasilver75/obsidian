
July 18, 2023 (~5 months after [[LLaMA]]) -- [[Meta AI Research]]
Paper: [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)


Technical details:
- Family: 7B, 13B, 70B (at every level, including both a base model and a chat model)
- Trained on: 2T tokens
- Context length: 4096 tokens
- RLHF with [[Proximal Policy Optimization|PPO]], [[Rejection Sampling]]
- "Ghost Attention"
- [[Grouped Query Attention]]
- Commercial use license
- [[LLaMA Guard]]

Abstract
> In this work, we develop and release ==Llama 2==, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called ==Llama 2-Chat==, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

### Preference Data Details:
- Collected ==binary comparisons, rather than other fancier feedback==... also information like *"significant better, better, slightly better, neglibgibly better/unsure"*
- Used ==multi-turn preferences==, where model responses are taken from different model checkpoints with varying temperature, to generate diversity between pairs.
- Focus collection on ==helpfulness and safety== (as opposed to *honesty*), using separate guidelines at data collection time for each vendor (safety is often a much more deceptive prompting style).
- The team added additional ==safety metadata== to the collection, showcasing whcih responses are safe from the models at each turn; when this is passed to the modeling phase, they don't include any examples where the chosen response was unsafe, while the other response *was* safe; "we believe safer responses will be better/preferred by humans, in the end."
- They don't detail additional metadata being logged, but it's likely they did, in order to identify potential bugs and data issues (confusing prompts, requiring tools to solve, etc.)
- Deployed ==iterative collection for distribution management==: "Human annotations were collected in batches on a weekly basis; as we collected more preference data, our reward models improved, and we were able to train progressively better versions for LLaMA 2-Chat"


![[Pasted image 20240417224708.png]]![[Pasted image 20240426014123.png]]

![[Pasted image 20240426020800.png]]
Above: The scale of preference data was impressive, likely costing $8M+ on dat alone, with way more turns than were often available at that time.

![[Pasted image 20240426021803.png]]
An awesome picture showing how the RLHF process shifts the generated texts towards higher rewards. The X axis is "Reward Model Score".

![[Pasted image 20240426132028.png]]
Showing the improvement on Helpfulness and Harmlessness benchmarks from both human and LM judges as we do multiple rounds of SFT and then RLHFing