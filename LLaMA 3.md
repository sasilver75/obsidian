April 18, 2024 (~9 months after [[LLaMA 2]]) - [[Meta AI Research]]
Paper: {Coming}

Refrences
- Video: [W&B Presentation from Director of Product @ Meta.ai re: LLaMA3](https://youtu.be/r3DC_gjFCSA?si=1zDwp3iildk6gL3W&t=367)
- Tweet: [Reaction by Andrej Karpathy](https://twitter.com/karpathy/status/1781028605709234613)
- Video: [Meta Engineering Interview: Understanding the Meta LLaMA 3 Tokenizer](https://youtu.be/Tmdk_H2WDj4?si=KKXIvT-FLC_wM69Z)

> "Working with a data vendor... I'm thinking they apply one method (of Rejection Sampling, DPO, PPO), collect new data, apply another method, collect new data.... and they try each method at every step, and whichever improves the model the most at a given step is what they use. This is really different from what we do in the open ecosystem, where he have static datasets where we're tweaking a bunch of knobs seeing what we can get out of our data. In industry labs, they're focused on continuing to push the *data itself;* the better they make their data, the better the alignment gets." - Nathan Lambert  [link](https://youtu.be/rDF7eFPeVto?si=dXbb4B88ljzHydIu&t=753)

---


Details:
- Family: 8B, 70B, 405B parameters
- Trained on: 15T tokens
	- The [[Chinchilla]] "compute-optimal" point for an 8B model would be to train it for ~200B tokens, meaning that this training is ~75X beyond that point.
- Context window: 8192 tokens (up from 4096 tokens in [[LLaMA 2]]) -- still quite small, but there may be fine-tunes that extend this shortly.
- Vocabulary size: 128k
Interestingly, these are all *dense* models (ie not [[Mixture of Experts]])

Meta AI product director, re: LLaMA3: "Really, the magic is in post-training; RLHF, PPO, DPO, Rejection Sampling, Instruction-Finetuning, Red-teaming re: CBRNE"