March 29, 2022
Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

Answers the question: "With a given compute budget, and ignoring inference costs, how do we choose between the number of parameters of our model and the number of tokens we train on?"

Abstract
> We investigate the ==optimal model size and number of tokens for training a transformer language model under a given compute budget.== ==We find that current large language models are significantly undertrained==, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. ==By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens==, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted ==compute-optimal model, Chinchilla==, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms [[Gopher]] (280B), [[GPT-3]] (175B), [[Jurassic-1]] (178B), and [[Megatron]]-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

==It can make sense to train a model that's *smaller* than Chinchilla optimal==, and train it for *longer* than Chinchilla would tell us, because if we're going to deploy the model at mass scale, we care *much more* about inference cost than we do training cost! - [Link](https://finbarr.ca/llms-not-trained-enough/)

GPT-4
> According to the Chinchilla findings, the optimal compute ratio implies that for every doubling of compute resources, one should increase the model size (parameters) by a factor of $2^{1/3}$​ (approximately 1.26) and the dataset size (tokens) by a factor of $4^{1/3}$​ (approximately 1.59). This implies a ratio of tokens to parameters that should increase as the compute budget grows, advocating for a larger emphasis on data scale relative to model size than previously practiced.

---
April 17, 2024 Update
People calling the replication of the paper into question: 
- (Jan 21, 2023): https://x.com/suchenzang/status/1616752482226671620
- (April 17, 2024): https://x.com/tamaybes/status/1780639257389904013

---