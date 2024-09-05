https://www.interconnects.ai/p/q-star

An emergency special; he information we need to understand what Q* is was right in front of us, but the memes are more fun than reality.

-----

My initial hypothesis, which I clearly labeled as a tin hat theory, was a vague merging of Q-learning and A* search. What I didn’t answer is, what is being searched over? My initial guess of searching over dialogue turns is almost certainly wrong due to infrastructure reasons I’ll touch on later.

As I’ve dug into this in more detail, I’ve become convinced that they are doing something powerful by **searching over language steps via [[Tree of Thoughts]] reasoning**, but it is much smaller of a leap than people believe. The reason for the hyperbole is the goal of linking large language model training and usage to the core components of Deep RL that enabled success like AlphaGo: self-play and look-ahead planning.

[[Self-Play]] is the idea that an agent can improve its gameplay by playing against slightly different versions of itself because it'll progressively encounter more challenging situations. In the space of LLMs, it's almost certain that the largest portion of self-play will look like AI Feedback, rather than competitive processes.

==Look-ahead Planning== is the idea of using a model of the world to reason into the future, to produce better actions or outputs. Variants are based on:
- ==Model Predictive Control== (MPC), which is often used on continuous states
- [[Monte-Carlo Tree Search]] (MCTS), which works with discrete actions and states

To understand how this links together, we need to cover recent results published from OpenAI and others that'll answer two questions:
1. ==How do we construct a representation that we can search over?==
2. ==How do we construct a notion of value over compartmentalized and meaningful language chunks (rather than over the entire completion)?==

If we had the answers to these, it would be more clear as to how we could use existing RL methods.

## Modular reasoning with LLMs: Tree of Thoughts (ToT) prompting
- Promoting techniques like "take a deep breath" and "think step by step" are now expanding into advanced methods for inference with parallel computation and heuristics (some of the fundamentals of search).
- [[Tree of Thoughts]] is really as simple as it sounds!
	- We prompt an LM to create a tree of reasoning paths that may or may not converge at a correct answer.
![[Pasted image 20240627003514.png|450]]
The innovations that make this click are:
- The *chunking of reasoning steps*
- The *prompting of a model to create new reasoning steps*

ToT seems like the first "recursive" prompting technique for improving inference performance.

With the reasoning trees, different methods can be applied to score each vertex, or to sample the final path.
- It can be based on things like minimum length to the most-agreed answer, or complex things that require external feedback, which points us in the direction of RLHF.


## Fine-grained reward labels in generation: Process Reward Models (PRM)
- Most RLHF is done to date by having the entire response from a language model, and then we give it a score.
	- ==to anyone with an RLHF background, this is disappointing, because it limits the ability for RL methods to make connections about the value of each sub-component of text.== (EG [[Value Function]]s of intermediate states)

Futures have been pointed to where this multi-step optimization comes at the level of multiple dialogue turns, but that's still far-fetched, due to the requirement of having humans or some prompt source in the loop.

This could be easily extended to a self-play-style dialogue, but it's hard to give an LLM goals that would translate to the self-play dynamics of constant improvement.
- Most of the things we want to do with LLMs are repetitive tasks without near-infinite ceilings on performance like the game of Go.

On the other hand, there's a type of LLM use case that naturally abstracts to contained chunks of text:  step-by-step reasoning (eg in math problems)

==Process Reward Models (PRMs)== have been a topic that he's heard about from a lot of RLHF folks off the record for the last 6 months; there's a lot of literature on these models, but very little on how to use them with RL.
- The core idea of a PRM is to assign some score to each step of reasoning, rather than to a complete message.
- An example is from the OpenAI paper "let's verify step by step"
![[Pasted image 20240627004632.png|300]]
This allows for finer-tuned generation with reasoning problems, by sampling over the maximum average reward, or other metrics, instead of just relying on one score.

Using [[Best-of-N Sampling]], essentially generating a bunch of of and using the one that scored the highest by the reward model (an inference-time cousin of [[Rejection Sampling]]), PRMs outperform standard RMs on reasoning tasks.

To date, most resources for PRMs show how to use them at inference time -- the true signal will come when this signal is optimized against training. To create the richest optimization setting, having the ability to generate diverse reasoning pathways for scoring and learning from is essential.
- [[Tree of Thoughts]] prompting can give diversity to generations, which a policy can learn to exploit with access to a PRM.


## Putting it together: What Q* could be
- It seems to be using PRMs to score Tree of Thoughts reasoning data, which is then optimized with offline RL. The "trajectory" seen by the RL algorithm is the sequence of reasoning steps, so we're finally doing RLHF in a multi-step fashion rather than contextual bandits.
	- If we can use AI to label every step with a score, instead of humans, that would be interesting. 
- Nato has heard that one or more of the big-tech players (Google, Anthropic, Cohere, etc.) are creating a pretraining-sized dataset from process supervision or [[Reinforcement Learning from from AI Feedback|RLAIF]]-like methods.


All of this said while the core ideas seem clear to me, implementing this takes levels of model whispering few poses. Distribution control, massive inference, and RL finickiness are well beyond my knowledge or experience.








