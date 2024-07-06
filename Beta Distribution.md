References:
- [Luis Serrano's Beta Distribution in 12 minutes](https://youtu.be/juF3r12nM5A?si=cH9S5v8JPNI9ZdUx)

A continuous probability distribution used to model random variables that represent probabilities or proportions.

![[Pasted image 20240706011854.png]]
- Distributions are from a one-armed bandit example with 3, 30, and 300 examples, where we won 1/3 of the time in each case.
- See in the top case that we always add one to our (eg) wins and losses (a, b). So we would "start" with Beta(1,1).

----

Beat Distributions model the probability (confidence?) of a probability!
Let's look at an example where we have three biased coins, none of which returns Heads with probability .5.
1. P(H) = .4
2. P(H) = .6
3. P(H) = .8
Now let's imagine that we close our eyes and select one of the coins randomly. The goal is to determine which of the coins we've selected, given our knowledge about their probabilities.

So let's say we flip it five times and receive $\{H,H,H,T,T\}$
- Which one do you think we grabbed?
	- It's mostly likely coin 2, but it could also be coins 1 or 3.
- The question is: For each coin, what's the probability that *this* is the coin we flipped?We just consider: What's the probability of coin producing that specific sequence of independent outcomes?
	1. $.4*.4*.4*.6*.6 = .0230$
	2. $.6*.6*.6*.4*.4 = .0346$ 
	3. $.8 * .8 * .8 * .2 * .2 = .0205$
The sum of these probabilities is $0.0781$, so we use it to normalize each coin probability to get the relative probability of each coin.
1. .0230/.0781 = .295
2. .0346/.0781 = .443
3. .0205/.0781 = .262

Let's do a similar example, but using 10 possible coins.
![[Pasted image 20240706100512.png]]
So which is the coin that's the most likely? Coin 7.
But what's the probability that it's each coin?
1. We calculate the probability of our sequence being generated, under each coin.
2. We normalize the probabilities by the sum of sequence probabilities.
![[Pasted image 20240706100715.png]]
So this plot starts looking more like a distribution.

Imagine now that we do the same experiment, but with thousands of coins.
- If a given coin has probability $p$ of giving heads, in this situation the probability that that sequence generated our outcome sequence of 7 heads and 3 tails, $p(x)$ is $p^7(1-p)^3$.

But w want to turn this into a probability distribution, meaning that all of the p(x)s must sum to zero.

![[Pasted image 20240706101418.png]]

So as an example
![[Pasted image 20240706101543.png]]
Notice that all have the mode at 0.7, but if we were to pick a random point under the curves, it's much more likely that the draw will be close to .7 on the one on the right.


