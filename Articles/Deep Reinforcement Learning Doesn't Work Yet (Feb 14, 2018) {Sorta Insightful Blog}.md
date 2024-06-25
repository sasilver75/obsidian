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


## Reinforcement Learning usually requires a Reward Function


## Reward Function design is difficult


## Even given a good reward, local optima can be hard to escape


## Even when Deep RL works, it might just be overfitting to weird patterns in the Environment


## Even ignoring generalization issues, the final results can be unstable and hard to reproduce


---

## But what about all the great things Deep RL has done for us?


## Given current limitations, *when* could deep RL work for me?


## A case study: Neural Architecture Search


## Looking to the Future

















