March 29, 2022 -- 7 months before [[ChatGPT]]
[[DeepMind]] - Authors include [[Arthur Mensch]], later founder @ [[Mistral]]
Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
#zotero 
Takeaway: Answers the question: "With a given compute budget, and ignoring inference costs, how do we choose between the number of parameters of our model and the number of tokens we train on?" Finds that the current LMs are significantly undertrained/overparametrized. Trains a new model, *Chinchilla (70B params, 1.4T token*s), that is actually compute-optimal; it outperforms Gopher, GPT-3, and others.

April 17, 2024 Update
People calling the replication of the paper into question: 
- (Jan 21, 2023): https://x.com/suchenzang/status/1616752482226671620
- (April 17, 2024): https://x.com/tamaybes/status/1780639257389904013

Replication Paper: "Chinchilla Scaling: A replication attempt" where they get slightly different numbers (really, it's one of the three Chinchilla techniques that didn't replicate)

*NOTE:* Chinchilla scaling laws describe compute-optimal *training*, of how to best use some amount of compute to minimize perplexity(?). In reality, you probably want *inference-optimal training,* where you take into account the fact that most of your costs of the model in the long run are going to be dominated by inference costs. This means you should do things like LLaMA 3, where you "overtrain" a smaller model on a large number of tokens.

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


# Non-Paper Figures
![[Pasted image 20240605160458.png]]
From CS685 Lecture 17
- For years, people used the Kaplan scaling laws; You can see that going off the Kaplan models resulted in earlier models being overparametrized (relative to Chinchilla-optimality), not prioritizing the dataset size.


![[Pasted image 20240605162547.png]]
All of the dots on each line use the same amount of compute, but vary in the number of parameters and data. Points on the left of each line are models trained with fewer parameters and more data, and points on the right are models trained with more parameters and less data. Each line is a different number of flops.
- Conclusion: The biggest model isn't always the best one -- actually, in NO case is the biggest model the best! In fact, the optimal model is somewhere in the middle! For each compute budget, you pick a model parameter size {somewhere in the middle} and a dataset size {somewhere in the middle} to minimize loss.

But a question is: Will these relationships hold? If we have a curve with a much larger compute budget?
- Authors take the minimum of each of these curves and plot the minimum as a function of dataset size and model size, yielding the following two graphs

![[Pasted image 20240605162821.png]]
They fit this curve to these numbers. But should we believe it? The authors choose to *show* it by creating a model, Chinchilla at the ~10^24 FLOPs point.

The [[Gopher]] model uses the older Kaplan et al. scaling model (280B params/300B tokens), and [[Chinchilla]], on the new scaling model, is 4x smaller (70B params/1.4T tokens). 

![[Pasted image 20240605163654.png]]
- How much will my loss increase if I reduce the size of my training set? 
	- The second term tells us that: What is the contribution of my dataset size to the model loss. Decreasing it increases the contribution towards loss.
- Here, Alpha and Beta are things that we're trying to learn/fit based on empirical training runs. AB,N,D as well.
- In the Chinchilla paper, they train a bunch of models with different compute budgets, and fit the model described above. The find the coefficients to be:
	- E = 1.69 
	- Alpha = .34
	- Beta = .28
	- A and B are both = 400

They end up getting:
L(Gopher) = 1.993
L(Chinchilla) = 1.936

The authors even say that, with this compute budget C, NO MODEL trained with 300B tokens could EVER be better than Chinchilla!
- But note that this was empirically fit, and depends on a bunch of things (hyperparameters, etc.).
- Also note that this is all about perplexity, not performance on the downstream tasks that we actually care about.

==Note that this doesn't cover inference costs at all==! Say you're an inference provider of some specific model, and you know that billions of people are going to use your model -- you want to have an overtrained, small language model (even though you have diminished returns on training cost), because the dominating inference costs are much smaller!

q: What about Distillation?
a: This doesn't really say anything about that.

q: What about repeating data?
a: This doesn't say anything about that, but there's controversy:
- Brown et al. 2022 says that repetition can lead to degradation in performance. "Performance of an 800M param model can be degraded to that of a 400M param model by repeating just .1% of the data!"
- The 2022 Galactica paper in contrast found that performance continued to ***improve*** on validation set and in-domain/out-of-domain benchmarks with multiple repeats (4.25 epochs tested) of the corpus.
	- ((I trust this one))