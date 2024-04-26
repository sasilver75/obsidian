#article 
Link: https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence

-----

[[Kullback-Leibler Divergence]] is a topic that crops up in a bunch of different places in information theory and machine learning, so it's important to understand it well... unfortunately, KL divergence is confusing at first pass!
- It's not symmetric, like we'd expect from most distance measures
- It can be unbounded as we take the limit of probabilities going to zero

But there are a lot of different ways that you can develop good intuitions for it!
This is an attempt to collate those intuitions, and try to identify the underlying commonalities between them.

---
# Summary

#### 1. Expected Surprise

$D_{KL}(P||Q)$  = how much ***==more surprised*==** you would expect to be when observing data with distribution $P$ , if you ***==falsely==*** believed the distribution is $Q$ (vs if you knew the true distribution)

#### 2. Hypothesis Testing

$D_{KL}(P||Q)$  = the amount of **==evidence==*** that we expect to get for $P$ over $Q$ in hypothesis testing, if $P$ were true.

#### 3. MLEs

If $P$ is an empirical distribution of data, $D_{KL}(P||Q)$ is minimized (over $Q$) when $Q$ is the ***maximum likelihood estimator*** for $P$.

#### 4. Suboptimal Coding

$D_{KL}(P||Q)$  = the number of bits we're wasting if we try and compress a data source with distribution $P$ using a code which is actually optimized for $Q$ (i.e. a code which would have minimum expected message if $Q$ were the *true* data source distribution)

((This is very similar to #1, to me -- which is fine!))

#### 5A. Gambling games -- Beating the house

$D_{KL}(P||Q)$  = the ==amount== (in log-space) we ==can win from a casino game, if we know the true distribution== is $P$ ==but the *house* **incorrectly believes*** it== to be $Q$.

#### 5b. Gambling games -- gaming the lottery

$D_{KL}(P||Q)$  = the amount (in log-space) we can win from a lottery if we know the winning ticket probabilities $P$ and the distribution of ticket purchases $Q$

((This of course seems to be the same thing as 5A))

#### 6. Bregman Divergence

$D_KL(P||Q)$ is in some sense ==a natural way of measuring how far $Q$ is from $P$,== if we're using the entropy of a distribution to capture how far away it is from zero (analogous to how $||x - y||_2$ is a natural measure of the distance between vectors $x$ and $y$, if we're using $||x||_2$ to capture how far the vector $||x||_2$ is from zero)


==The common theme for most of these:==

$D_KL(P||Q)$  -> [[Kullback-Leibler Divergence]] is a measure of how much our model $Q$ differs from the true distribution $P$. In other words, we care about how much $P$ and $Q$ differ from eachother ***in the world where P is true***, which explains why KL-divergence is not symmetric!


---

# 1. Expected Surprise

For random variable $X$ with probability distribution $P(X = x) = p_x$ 
The **surprise** is defined as $I_p(x) = -ln(p_x)$ 
This is motivated by some simple intuitive constraints that we would like to have on *any* notion of surprise!

==Surprise Rules==
1. An event with probability 1 has *no surprise*
2. *Lower-probability events* are strictly *more surprising*
3. Two independent events are exactly as surprising as the sum of those events' surprisal when independently measured.


From this, we have another way of defining [[Entropy]] -- as the expected surprised of an event!

==Surprise==
$H(X) = -\sum_x{p_xln(p_x)} = \mathbb{E}_P[I_P(X)]$ 
(Entropy is the expected surprise!)

((In my opinion, this is skipping a step of derivation, as to why we use -plogp. The [[Visual Information Theory (Oct 2015) {Chris Olah's Blog}]] blogpost has some information on this. By the way, you can use $log$ too, I think it's some notational thing))


Now, suppose we (erroneously) believed that the true distribution of $X$ to be $Q$, rather than $P$!
Then the expected surprise of our model (taking into account that the true distribution of $P$) would be:

==Cross Entropy==
$\mathbb{E}_P[I_Q(X)] = -\sum_x{p_xln(q_x)}$ 
((Missing a derivation here, I think 😄 or at least intuition as to why *that* part gets the q, rather than the p))


And we now find that:

==Kullback Leibler Divergence==
$D_{KL}(P||Q) = \sum_x{p_x(ln(p_x) - ln(q_x))} = \mathbb{E_P}[I_Q(X) - I_P(X)]$ 

In other words, ==KL Divergence is the difference between the *expected surprise of YOUR model, and the expected surprise of the CORRECT model* (i.e. the model where you *know* the true distribution P).== The further apart your $Q$ is from the true $P$, the worse the model $Q$ is for $P$ -- i.e., the more surprised ti should expect to get by reality.

This also explains why $D_{KL}(P||Q)$ isn't symmetric
- It blows up when $p_x >> q_x$ , but not the inverse. In the former case, your model assigns very low probability to an event which might happen quite often., hence your model is very surprised by this.
	- The inverse doesn't have this property, and there's no equivalent story you can tell about how your model is frequently very surprised.

---

# 2. Hypothesis Testing
- Suppose you have two hypotheses: 
	1. A null hypothesis $H_0$ which says that $X \sim P$ 
	2. An alternative hypotheses $H_1$ saying that $X \sim Q$ 

Let's suppose that the *null hypothesis* is actually true!

![[Pasted image 20240413215220.png]]

-------

# 3. MLEs (Maximum Likelihood Estimation)

This one is a bit more maths-heavy than the others, so YMMV on how enlightening it is...
![[Pasted image 20240413215306.png]]

---

# Suboptimal Coding

- Source coding is a huge branch of information theory, and I won't go through all of that in this post. There are several online resources that do a good job of explaining it (See Chris Olah)

Recap of it:
> If you're trying to transmit data from some distribution over a binary channel, you can assign particular outcomes to strings of binary digits in a way which minimizes the expected number of digits you have to send.
> You want to use shorter codes for outcomes that occur with more probability.
> In the limit for a large number of possible values for X, the optimal code will represent outcome $x$ with a binary string of length $L_x = -log_2({p_x})$ 

From this, the intuition for KL divergence pops neatly out:

Suppose you *erroneously* believe that $X \sim Q$, and you designed an encoding that would be optimal for this case (meaning, it's really optimized for Q, the distribution that the true distribution $X$ *is not!*)

The expected number of bits you'll have to send per message is:

$-\sum_x{p_xlog_2(q_x)}$ 
((Why?))

We can immediately see that ==KL-divergence is== (up to a scale factor) the ==*difference in the expected number of bits per events you'll have to send with this suboptimal code, versus the number you'd expect to send if you knew the true distribution and could construct the optimal code!*==

The further apart P and Q are, the more bits that you're wasting on average by not using the optimal code (the one designed for $P$, rather than $Q$).


-----

# 5A. Gambling Games - Beating the House

- Suppose you can bet on the outcome of some casino game, like a version of roulette wheel with nonuniform probabilities.
- First, imagine the house is fair, and pays you $1/p_x$ times your original bet if you bet on outcome $x$ (this way, any bet has zero expected value).
	- Because the house knows exactly what all the probabilities are, there's no way for you to win money in expectation!
	- If you bet $c_x$ , you expect to get $p_x * \frac{c_x}{p_x} = c_x$ in return, which is ... no return!

- Now, imagine that the house *doesn't* know the true probabilities $P$, but *you do!*
	- The house's mistaken belief is $Q$, so they pay people $1/q_x$ for event $x$ even though this actually has probability $p_x$ .
![[Pasted image 20240413223442.png]]

---

# 5B. Gambling Games - Gaming the Lottery

... Skipping because it's very similar to 5A ...

----
# 6. Bregman Divergence

Skipping this because it's quite mathematically difficult, even to the author.


----

# Final Thoughts


Recapping, these, we find that if $D_{KL}(P||Q)$ is large, this indicates that:
1. Your model $Q$ will be very surprised by reality $P$
2. You expect to get a lot of evidence in favour of hypothesis $P$ over $Q$, if $P$ is true
3. $Q$ is a poor model for observed data $P$ 
4. You would be wasting a lot of message content if you tried to encode $P$ optimally, while falsely thinking that the distribution were $Q$ 
5. You can make a lot of money in betting games where other people have false beliefs $Q$, but you know the true probabilities $P$

These all have in common:
> $D_{KL}(P||Q)$ is a measure of how much our model $Q$ differs from the true distribution $P$. In other words, we care about how much $P$ and $Q$ differ from eachother *in the world where $P$ is true*, which explains why KL-divergence is not symmetric.

To put this last point another way, $D_{KL}(P||Q)$  "doesn't care" when $q_x >> p_x$ (assuming both probabilities are small), because even when our model is wrong, reality doesn't frequently show us situations in which our model fails to match reality. But if $p_x >> q_x$, then the outcome $x$ will occur more frequently than we'd expect, consistently surprising our model and thereby demonstrating the model's inadequacy.





















