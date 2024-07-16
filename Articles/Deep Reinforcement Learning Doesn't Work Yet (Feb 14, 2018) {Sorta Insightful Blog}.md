https://www.alexirpan.com/2018/02/14/rl-hard.html
For context, since this article, the author iirc has mentioned being exciting about some uses of RL in industry that they're beginning to see.
I'm not yet knowledgable to know yet which of these claims still hold in 2024 or not. This article was before AlphaStar, even.

----

On Facebook, the author made the claim
> Whenever someone asks me if RL can solve their problem, I tell them that it can't. I think this is right about 70% of the time.

Deep Reinforcement Learning is surrounded by mountains and mountains of hype.
- It's an incredibly general paradigm, and, in principle, a robust and performant RL system should be great at everything.
- Merging this paradigm with the empirical power of deep learning is an obvious fit!

But it doesn't really work yet 🥲

If the author didn't believe in RL, though, they wouldn't be working on it!

The beautiful demos of learned agents that you might see online hide all of the blood, sweat, and tears that go into creating them.
- New entrants often underestimate deep RL's difficulties!
- Without fail, the "toy problem" isn't going to be as easy as it looks! It's important to set realistic research expectations.

==In this post, the goal is to explain why deep RL doesn't work, cases where it does work, and ways the author can see it more reliably in the future.== He just wants new entrants to know what they're getting into.

The post is structured to go from pessimistic to optimistic.

Here are some of the failure cases of deep RL

## Deep Reinforcement Learning Can be Horribly Sample Inefficient
- The most well-known benchmark for Deep RL is Atari, shown in the now-famous [[Deep Q-Networks]] (QN) paper -- if you combine [[Q-Learning]] with reasonably-sized NNs and some optimization tricks, you can get human/superhuman performance in several Atari games.
- Can you estimate how many frames a SoTA DQN needs to reach a performance?
	- ![[Pasted image 20240624195422.png|300]]
	- It seems that the best method here passed the median human score after ~18 million frames (at 60/second in Atari, that'd be 300,000 seconds)

## If you just care about final performance, many problems are better solved by other methods
- For purely getting good performance, deep RL's track record isn't that great, and consistently gets beaten by other methods (even many that are computed in near real-time, online, with no offline training).
- Planning against a model (Which we often don't have a in "real" RL) can help a lot.
	- Methods like [[Model Predictive Control|MPC]] and [[Monte-Carlo Tree Search|MCTS]] can easily outperform [[Deep Q-Networks|DQN]] and similar fancy techniques. This is perhaps unfair, because DQN does no search, whereas MCTS gets to perform search against a ground-truth model.
	- Model-free learning forces you to use tons of samples to learn things that could have been hardcoded.

## Reinforcement Learning usually requires a Reward Function
- RL assumes the existence of a reward function; this is usually either given, or is hand-tuned offline and kept fixed over the course of learning. Most RL approaches treat the reward as an oracle.
- ==Importantly, for RL to do the right thing, your reward function must capture *exactly* what you want -- and I mean *exactly.* RL has an annoying tendency to overfit to your reward, leading to unexpected behavior.==

## Reward Function design is difficult
- Making *just any* reward function isn't difficult, but making one that encourages the behaviors that you want and discourages the ones you don't, while *still being learnable* is difficult!
- Shaped rewards from users can bias learning, leading to behaviors we don't want (eg the boat reward-hacking behavior in the OpenAI Atari demo)
- RL algorithsm fall along a continuum of knowing more or less knowledge about their environment. The broadest category is [[Model-Free]], which is basically black-box optimization; these are only allowed to assume that they're in some MDP.
	- If rewards are sparse, a +1 reward might be received at termination, even if it's not coming for the right reasons.
- (Example of salesforce using RLR to optimize ROGUE for summarization in 2017, which resulted in good ROGUE scores, but the summaries sucked)
> Button was denied his 100th race for McLaren after an ERS prevented him from making it to the start-line. It capped a miserable weekend for the Briton. Button has out-qualified. Finished ahead of Nico Rosberg at Bahrain. Lewis Hamilton has. In 11 races. . The race. To lead 2,000 laps. . In. . . And.

## Even given a good reward, local optima can be hard to escape
- Reward hacking is the exception, in reality -- more commonly, a poor local optima results from getting the explore-exploit tradeoff wrong.

There are several intuitively pleasing ideas for addressing this - intrinsic motivation, curiosity-driven exploration, count-based exploration, and so forth. However, as far as I know, none of them work consistently across all environments.

==I’ve taken to imagining deep RL as a demon that’s deliberately misinterpreting your reward and actively searching for the laziest possible local optima. It’s a bit ridiculous, but I’ve found it’s actually a productive mindset to have.==


## Even when Deep RL works, it might just be overfitting to weird patterns in the Environment
- The upside of reinforcement learning is that if you want to do well in an environment, you’re free to overfit like crazy. The downside is that if you want to generalize to any other environment, you’re probably going to do poorly, because you overfit like crazy.
	- You have to ask yourself; do I really need generalization, or is my test set literally my training set? It depends if you're playing Atari or building a self-driving car.
- DQN can solve a lot of the Atari games, but it does so by focusing all of learning on a single goal - getting really good at one game.

This seems to be a running theme in multiagent RL. When agents are trained against one another, a kind of co-evolution happens. The agents get really good at beating each other, but when they get deployed against an unseen player, performance drops. I’d also like to point out that the only difference between these videos is the random seed. Same learning algorithm, same hyperparameters. The diverging behavior is purely from randomness in initial conditions.
- That being said, there are some neat results from competitive self-play environments that seem to contradict this. [OpenAI has a nice blog post of some of their work in this space](https://blog.openai.com/competitive-self-play/). Self-play is also an important part of both AlphaGo and AlphaZero.

## Even ignoring generalization issues, the final results can be unstable and hard to reproduce

Supervised learning is stable. Fixed dataset, ground truth targets. If you change the hyperparameters a little bit, your performance won’t change that much. Not all hyperparameters perform well, but with all the empirical tricks discovered over the years, many hyperparams will show signs of life during training. These signs of life are super important, because they tell you that you’re on the right track, you’re doing something reasonable, and it’s worth investing more time.

Currently, deep RL isn’t stable at all, and it’s just hugely annoying for research.

(Story of spending 6 months at Google Brain to reproduce some results)

---

## But what about all the great things Deep RL has done for us?
- I tried to think of real-world, productionized uses of deep RL, and it was surprisingly difficult. I expected to find something in recommendation systems, but I believe those are still dominated by [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) and [contextual bandits](https://research.yahoo.com/publications/5863/contextual-bandit-approach-personalized-news-article-recommendation).
- The way I see it, either deep RL is still a research topic that isn’t robust enough for widespread use, or it’s usable and the people who’ve gotten it to work aren’t publicizing it. I think the former is more likely.


## Given current limitations, *when* could deep RL work for me?
Works good where:
- It's easy to generate nearly unbounded amounts of experience (eg Atari, Go, Chess)
- The problem can be simplified into an easier form (don't solve the full thing end-to-end if you don't have to -- use external information/tools to help your bot understand state)
- There's a way to introduce self-play into learning (eg AlphaGo, AlphaZero, SSBM Falcon bot)
- When there's a clear way to define a learnable, ungameable reward (+1 for win, -1 for loss, accuracy on the test set).
- If reward has to be shaped, it should at least be rich (rather than sparse). Reward signals that come quick and often (damage dealt, damage taken)
## A case study: Neural Architecture Search
According to the initial [ICLR 2017 version](https://arxiv.org/abs/1611.01578), after 12800 examples, deep RL was able to design state-of-the art neural net architectures.
As mentioned above, the reward is validation accuracy. This is a very rich reward signal - if a neural net design decision only increases accuracy from 70% to 71%, RL will still pick up on this.

## Looking to the Future
I see no reason why deep RL couldn’t work, given more time. Several very interesting things are going to happen when deep RL is robust enough for wider use. The question is how it’ll get there. Ideas include:
- **Local optima are good enough**
- **Hardware solves everything**
- **Add more learning signal**
- **Model-based learning unlocks sample efficiency**
- **Use reinforcement learning just as the fine-tuning step**
- **Reward functions could be learnable**
- **Transfer learning saves the day**
- **Good priors could heavily reduce learning time**
- **Harder environments could paradoxically be easier**
















