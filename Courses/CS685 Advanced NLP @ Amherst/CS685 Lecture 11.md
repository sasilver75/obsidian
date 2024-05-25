Link: https://www.youtube.com/watch?v=Iw2gHYRF_TA&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=11
# Topic: LLM Alignment and RLHF
![[Pasted image 20240524000225.png]]
Limitations of Instruction tuning
1. We only learn from *positive examples* ("act like this"), but not from negative examples ("Here's an example of a bad output; don't do this.")
2. Some prompts (e.g. creative ones) have *many* acceptable outputs, but we can only train (positively) on one of them (at a time), and we can't possibly think of all of them! Next-token-prediction here isn't a good proxy when there are multiple ways of responding.
3. How to encourage abstaining in the case where the model *doesn't know about something?* We would have to know what the model does/doesn't know for all things -- yikes!
4. NTP != Human Preferences!



![[Pasted image 20240524000716.png]]
We use a SFT'd language model and prompt, and sample various responses. We have a human (or AI) produce preference judgements/rankings over the responses. It's expensive to do this, because y1/y2/y3 could be long and complicated, requiring specialized knowledge (eg "Write an essay about NLP in 2004").
- The thumbs up/down feature on ChatGPT is another form of getting this feedback.

So if human annotator rankings are very expensive, is it possible to create some sort of model that, given a prompt and response, can *predict human evaluator's ratings of it?* This is called a [[Reward Model]]

The input to a reward model is a prompt $x$ and a generation $y_i$ , produce a scalar score rating.

OpenAI uses a [[Bradley-Terry Model]] of preferences, where: 
![[Pasted image 20240524001632.png]]
If this is our preference model, how do we get a ==loss function== from this?

We know from working with the Softmax function that our favorite loss is the negative log likelihood; we want the probability of the winning output to be 100%, and the probability of the losing output to be 0% under this probabilistic framework; we do this by maximizing the negative log likelihood, as per the original language modeling loss.

If we do that to our probability and simplify it (as per the instruct GPT paper), we get:

![[Pasted image 20240524002040.png]]
In the end, we have a loss function for training a reward model outputting this scalar score, but we can see that the reward is computed using both the winning and losing output
- So we need to pass both the winning and losing through the reward model, get their scores, and then calculate the loss here.


We need to use a pretty capable model for our reward models, because the task of discriminating which response is better is (in the worst case) just as hard as *generating* that really good response!
- For this reason, we often start with the SFT LLM itself, with the head decapitated and replaced to perform scalar regression for our reward.

Now, using our reward model, for any given prompt, we can pass it into our SFT LLM, generate a number of outputs, and pass each of them individually into the reward model (which we've just trained) to get a score.

![[Pasted image 20240524003117.png]]

So now we've scaled our ability to assign preference scores to our LLM's generations, but how do we use this to actually align our LLM to human preferences?

Two methods we'll talk:
1. Best-of-N sampling (a [[Rejection Sampling]])
	- At *actual* inference time, generate N samples for a given prompt, score each sample using our reward model, and then choose the sample with the highest reward. 
	- This strategy is simple! But it's quite expensive; we have to do (eg) 16, or 32 forward passes every time we want to generate a response! We'd much more like a method that only requires us (at later inference time) to only do a single forward pass.
	- (It's been shown that this method is basically just as good as RLHF on benchmarks, interestingly)
2. Just finetune the LLM to maximize $p(y_w | x)$
	- Use the reward model to identify the best sample in the way described above, and just finetune on that. 
	- It's a pretty good strategy, and is simple, but unfortunately it still only works from positive examples, without negative examples that would be very useful to have.
3. Use Reinforcement Learning to increase the probability of the best sample $p(y_w|x)$, and decrease the probability of a losing sample $p(y_w|x)$, where the amounts are functions of the rewards $R(x, y_w)$ and $R(x, y_l)$.
	- In this example, we learn from both good and bad samples by learning to maximize reward in our RL scenario.
	- We only receive a reward after generating a full sequence (not after generating a single token). 
	- Our reward only tells us if the response was good or bad, not "Oh, at this fifth token here, you said X, but you should have said Y."
		- Here, we get into the various RL algorithms, and how they tackle the [[Credit Assignment Problem]].
	- We're not going to talk about [[Proximal Policy Optimization|PPO]] too much in this class.
	- But assume that we have some $\pi_{ref}$ from our SFT LLM checkpoint, and some $\pi$ that's our current policy model (initialized to $\pi_{ref}$). We're going to update the parameters of $\pi$ and keep the parameters of $\pi_{ref}$ frozen -- we don't want the parameters of our $\pi$ to move too far away from the parameters of our initial SFT model (which we know has a lot of powerful instruction-following ability already; we don't want to forget too much of that!). 
		- We need to keep these two models in our GPU memory, as well as our reward model, which *also* was initialized from our SFT checkpoint.
		- ==So even if we want to do RLHF with LLaMA2-7B, we have to keep *three* 7B checkpoints in our GPU if we want to accomplish this. This blows up the amount of VRAM you need to do alignment by RLHF.==
	- To find the policy $\pi$ that maximizes the expected reward given some prompt $x$  --> (meaning $R(\pi(x)$), but without diverging too far from our origin policy, we'll introduce a [[Kullback-Leibler Divergence|KL-Divergence]] penalty:

![[Pasted image 20240524012524.png]]
 The KL divergence is applied at the token level; for every position in the sequence, we compute this penalty and then average it out over all tokens in the sequence.

We optimize this all using the [[Proximal Policy Optimization|PPO]] (Schulman, 2016)
Gemeni didn't use PPO, instead using a much simpler algorithm [[REINFORCE]] (the original [[Policy Gradient]] algorithm, invented in 1992).

The most commonly used algorithm however for academics these days is [[Direct Preference Optimization]] (DPO).



Let's go back and have a high-level overview of the process:
1. Supervised Fine Tuning
2. Reward Modeling (derived via (eg) Bradley-Terry preference model)
3. Reinforcement Learning/Optimization

![[Pasted image 20240524013635.png]]


