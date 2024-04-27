March 29, 2022 -- 7 months before [[ChatGPT]]
[[DeepMind]] - Authors include [[Arthur Mensch]], later founder @ [[Mistral]]
Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
#zotero 
Takeaway: Answers the question: "With a given compute budget, and ignoring inference costs, how do we choose between the number of parameters of our model and the number of tokens we train on?" Finds that the current LMs are significantly undertrained/overparametrized. Trains a new model, *Chinchilla (70B params, 1.4T token*s), that is actually compute-optimal; it outperforms Gopher, GPT-3, and others.

April 17, 2024 Update
People calling the replication of the paper into question: 
- (Jan 21, 2023): https://x.com/suchenzang/status/1616752482226671620
- (April 17, 2024): https://x.com/tamaybes/status/1780639257389904013
----


Notes
- Whereas the previous Kaplan Scaling Laws paper found that (if 10x compute budget, then 5.5x the number of parameters, and 1.8x the number of training tokens), this paper instead finds that parameters and tokens should scale at roughly the same rate. Additionally, the Kaplan paper seems to have considered models up to 1B in size, whereas the Chinchilla paper considers models up to 16B in size.
- For Gopher, they predict that the compute-optimal version should be 4 times smaller, trained on 4 times more data.
- The Chinchilla paper has three different approaches (see figure 1), but all three predict that current models should be smaller and trained longer (on more tokens) than is currently done.
	- Approach 1: Fix the model size and vary the number of training tokens: Extract an estimate of the minimum loss achieved for a given number of FLOPs.
	- Approach 2: Vary the model size for a fixed set of 9 different training FLOP count: For a given FLOP budget, what's the optimal parameter count?
	- Approach 3: Fit a parametric loss function; model all final losses from experiments in Approach 1/2 as a parametric function of parameters and tokens.
- Finds that setting the learning rate schedule to approximately match the number of training tokens results in the best loss, regardless of model size.


Abstract
> We investigate the ==optimal model size and number of tokens for training a transformer language model under a given compute budget.== ==We find that current large language models are significantly undertrained==, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. ==By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens==, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted ==compute-optimal model, Chinchilla==, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms [[Gopher]] (280B), [[GPT-3]] (175B), [[Jurassic-1]] (178B), and [[Megatron]]-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

==It can make sense to train a model that's *smaller* than Chinchilla optimal==, and train it for *longer* than Chinchilla would tell us, because if we're going to deploy the model at mass scale, we care *much more* about inference cost than we do training cost! - [Link](https://finbarr.ca/llms-not-trained-enough/)

GPT-4
> According to the Chinchilla findings, the optimal compute ratio implies that for every doubling of compute resources, one should increase the model size (parameters) by a factor of $2^{1/3}$​ (approximately 1.26) and the dataset size (tokens) by a factor of $4^{1/3}$​ (approximately 1.59). This implies a ratio of tokens to parameters that should increase as the compute budget grows, advocating for a larger emphasis on data scale relative to model size than previously practiced.

![[Pasted image 20240427124836.png]]
Above: They had three different methods (with some epistemic uncertainty, I guess?), but they all made vaguely similar conclusions -- that Kaplan's model was estimating that models should be larger than the Chinchilla team thought they should be.

![[Pasted image 20240427125113.png]]
Above: Most models were trained on 300B tokens at the time just because *that's what GPT-3 did*, and everyone was trying to replicate/beat GPT-3!



