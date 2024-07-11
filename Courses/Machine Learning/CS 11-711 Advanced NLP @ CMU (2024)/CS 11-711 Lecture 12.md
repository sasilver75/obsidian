# Topic: Learning from Human Feedback
https://www.youtube.com/watch?v=NX0l1M0NWPM&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=12

---

Up until now, we've been doing Maximum Likelihood Estimation, where we train our models to exactly predict the sequences that we have in our training dataset.

![[Pasted image 20240617123639.png]]
We minimize the negative log likelihood of producing the training sequence.

But is this what we want from our language models?

### Problems with Negative Log Likelihood

Problem 1: Some mistakes are worse than others
- We want good outputs, and some predictions hurt more than others, so we'd like to penalize them appropriately!
	- e.g. instead of "Please send this package to Pittsburg"
		- "Please send `a` package to Pittsburgh"
		- "Please send this package to `Tokyo`"
		- "`Fucking` send this package to Pittsburgh"
- From the point of view of Maximum Log Likelihood, these are all equally good/bad!

Problem 2: The "Gold Standard" in MLE can be bad!
- Corpora are *full* of outputs that we wouldn't want a LM to reproduce!
	- Toxic comments on Reddit
	- Disinformation
	- Translations from old MT systems (Lots of the content on the intent is translated content, much from when translation systems weren't nearly as good as they are now.)

Problem 3: [[Exposure Bias]]
- MLE training doesn't consider the necessity for generation; it relies on gold-standard context -- which are often good generations...
- Exposure bias: The model is not exposed to mistakes during training, and cannot deal with them at test time. In a gold-standard output, if a word has appeared previously, that word is more likely to happen next: "I am going to Pittsburg.", you're likely to talk about Pittsburg again in the near future; you end up with models saying "I am going to Pittsburg. I am going to Pittsburg. I am going to Pittsburg."

### Measuring how "Good" an Output Is
- ==Objective assessment== (for some tasks)
	- Having an annotated "correct" answer and matching against it; e.g. solving math problems, answering objective questions.
- ==Human-subjective annotation==
	- We might have a source input, several hypotheses, and we ask a human annotator to score those hypotheses. e.g. open-ended generation
	- [[Direct Assessment]] (a term from MT): Give a score directly to an output
		- Can be hard to get consistency for ratings when doing this; raters often aren't well-calibrated or correlated.
		- Often we assign score based on desirable traits:
			- Fluency: How natural is the output
			- Adequacy: In translation, how well does the output reflect the input semantics?
			- Factuality: Is the output factually-entailed?
			- Coherence: Does the output fit coherently in the discourse?
			- etc.
	- Preference Ratings/[[Pairwise Ranking]]
		- Have 2+ generations, and ask a human which is better than the other, or if they're tied.
		- +: Can be easier and more intuitive than direct assessment.
		- -: You can't tell if all systems are really good or really bad (maybe you have two shitty generations).
	- There generally are a limited number of things that humans can compare before their calibration falls apart. A typical way around this (eg used by [[ChatBot Arena]]) is to use something like an ELO or TrueSkill ranking (created for Chess, Games), where they battle against eachother pairwise in multiple matches, and these ranking algorithms give you a score about how good each of the players are.
	- Human Feedback: ==Error Annotation==
		- Using humans to annotate individual errors within the outputs, as more fine-grained feedback... but can be very time-consuming.
- ==Machine prediction of human preferences==
	- Machine prediction of human preferences is a more scalable way of providing feedback.
	- Variously called "automatic evaluation" (MT), "reward models" (RL, Chatbots)
	- Embedding-based Evaluation
		- Some unsupervised calculation based on embedding similarity between a reference and your generation (eg [[BERTScore]]). 
			- "The weather is cold today" and "It is freezing today" will both embed to similar space, though they don't have much lexical overlap.
	- Regression-based Evaluation
		- Supervised training of an embedding-based regressor. Calculate a whole bunch of human judgements (direct assessment or pairwise judgements).
			- Direct: Use a regression-based loss (eg MSE)
			- Pairwise: Use a ranking-based loss
		- [[COMET]] in MT is an example of one of these that works quite well.
	- QA-based evaluation
		- Asking a language model how good the output is ([[LLM-as-a-Judge]], it seems?)
			- eg GEMBA (Kocmi and Federmann 2023): They basically just ask GPT-4 to score a translation.
			- Lots of problems with biases of these models (length bias, positional bias, self-enhancement bias), as well as [[Goodhardt's Law]] (exploiting of the metric)
			- AutoMQM: You can also ask about fine-grained mistakes (where you don't ask the model to give a specific score, but instead ask the model to identify fine-grained mistakes in the model.)
	- When we have human evaluations and automatic evaluations, we typically calculate some comparison metrics between them (eg [[Pearson Correlation Coefficient]] or [[Kendall Rank Correlation Coefficient]]). If authors haven't done this sort of meta-evaluation, be suspicious.
- ==Use in another system/in a downstream task==
	- Intrinsic evaluation: Evaluating the quality of the output itself
	- Extrinsic evaluation: Evaluating output quality by its utility in downstream tasks.
		- Example: Evaluate LLM summaries through their Question Answering accuracy.
			- But what if you QA model is bad? How to disentangle the effect of the LM summary vs the QA model?


----

## Reinforcement Learning Basics: Policy Gradients


What is RL?
- Learning where we have:
	- An environment X
	- Ability to make actions A
	- Get a reward R
- Rewards may be sparsely disbursed (credit assignment problem), and we model probabilistic transition functions from states s->s'|a.

One of the interesting things about the reward R is that you can be creative about how you shape reward; you can come up with one that really matches what you want out of your model.

But why reinforcement learning in NLP?
- We might have a typical RL scenario: A dialogue where we can make responses, and will get a delayed reward (thumbs up, down) at the end. An example of this might be call centers (if you resolve a ticket without going to a human operator, your system gets a reward; If the person has to use a human, you get a small negative reward; If the person yells at you and hangs up, you get a really negative reward)
- We have many latent variables (eg chains of thought) where we decide the latent variable, and then get a reward based how the latent variables effect the output.
	- The CoT itself might not actually be "good"; you might have a bad CoT and still get the right answer, which makes it an RL problem ((?))
- You might have a sequence-level evaluation metric such that we can't optimize it without first generating a whole sequence.


Self-Training
- We sample or argmax according to the current model, and then use this sample (or samples) to maximize likelihood.
	- Does this seem like a good idea?
		- If you don't have any access to any {generation?} that looks good, this will end up with us optimizing towards good *and* bad outputs.
		- Nonetheless, self-training does improve your accuracy somewhat in some cases (eg when your model is more often right than wrong).


[[Policy Gradient]]/[[REINFORCE]]
- Add a term that scales the loss by the reward
	- If you can get a reward for each output, then instead of doing self-training entirely by itself, we multiply it by a reward, allowing us to increase the likelihood of things that get a high reward and decrease the likelihood of thing that get a low reward.

Credit Assignment for Rewards
- How do we know which action led to the reward?
- Best scenario is that we get immediate reward after each action... this is completely infeasible, though! In reality, we usually only get a reward at the end of a full rollout.
- Often, we assign exponentially-decaying rewards for future events to take into account the time delay between action and reward.
	- (Then he says something about it being the case that it's common to often just split up the future reward among the previous time steps, without decay)

Stabilizing Reinforcement Learning
- It can be unstable (like other sampling-based methods) and hard to get to work properly!
- It's particularly unstable when using bigger output spaces (eg the large vocabulary of a language model).
- Solutions
	- It's common to start training (==pre-training) with MLE== (maximum likelihood estimation) and then switch over to RL; this works in situations where we can run MLE.
	- ==Regularization to an Existing Model==: Having an existing model, and preventing it from moving too far away. If we assign a negative reward to toxic language utterances.... we want to keep our language model close enough to being a competent language model, while also removing toxic utterances; what's the least work we can do to accomplish this?
		- KL Regularization: Our loss has two terms, with the first being a term that improves reward, and the second being a term that penalizes distance from the original policy.
		- Proximal Policy Optimization: (Explanation unclear)
- ![[Pasted image 20240617142216.png]]
All of these are implemented in libraries like [[TRL]] from HuggingFace.


Adding a Baseline
- If we get a reward of 0.8 for a very easy sentence, when the baseline says that we should be getting 0.95, that's actually quite bad! Similarly for the inverse. Can we predict a-priori how difficult an example is, and adjust our reward based on that?
![[Pasted image 20240617142513.png]]
We do this by having a separate baseline model that predicts this, and adjust appropriately.
- Baseline can be anything; the purpose of it is to decrease variance
	- Predicting the final reward using a model that doesn't look at the answer you've provided, only at the input ((?)) (Ranzato et al. 2016) You can have one baseline per sentence, or one baseline per output action.
	- Use the mean of the rewards in the batch as the baseline. (Dayan 1990)



Contrasting Pairwise Examples
- We can learn directly from pairwise (human) preferences, which provides more spability. E.g. [[Direct Preference Optimization|DPO]]
- Basically calculates a ratio of the probability of the new model to the old model, but it upweights the probability for a good output, and downweights the probability of a bad output.
- ![[Pasted image 20240617142904.png]]
- DPO is vey similar to PPO, in the sense that it's using these ratios... but the disadvantage of DPO is that it requires pairwise judgements.

Because RL is going to have higher variance than MLE because we're doign sampling and other things like this... one very simple thing you can do is just increase the number of rollouts that you do before an update (eg increasing batch size):
![[Pasted image 20240617143121.png]]
We can even save many previous roll-outs and re-use them as we update parameters!





