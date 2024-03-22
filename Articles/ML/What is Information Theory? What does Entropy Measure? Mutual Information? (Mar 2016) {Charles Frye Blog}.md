#article 
Link: https://charlesfrye.github.io/stats/2016/03/29/info-theory-surprise-entropy.html

-----

Information theory provides a set of mathematical ideas and ==tools for describing uncertainty about the state of a random variable== that are complementary to standard methods from probability theory.

Among the tools of information theory are [[Entropy]] and [[Mutual Information]]

In short, the ==entropy== of a random variable is an ==average measure of the *difficulty in knowing the state of that variable.*==

The ==mutual information==, on the other hand, tells us ==how much more we know about the state of two random variables when we think about them *together* instead of considering them separately==.

![[Pasted image 20240313150819.png]]
Above: ((In other words, Probability is the calculus of Gambling 😄))

# A problem
- In the world of Aristotelean Classical logic, the comparison of competing claims is simple -- if two claims are well-formed, their truths can be ascertained -- balls are red or not -- they will hit the floor when dropped, or they will not.
	- But there are many claims we want to make that *cannot* be so cleanly logically deduced!
	- *Inductive Reasoning* (the sun came up yesterday, and so it will tomorrow) is exactly such a claim.

These claims (that are neither definitely positively true or definitely false) arise frequently!
- Producing weather reports, catching the bus, predicting the outcome of elections, interpreting experimental results, and betting on sports games.

# Surprise!
==The goal in all cases is to *guess* something that *we don't* or *can't* know directly, like the future(!), on the basis of these *we do know*, like the present+past.==

In predicting the unknown future, we seek to avoid ==surprise!==
- If I have the correct weather forecast, I'm not surprised when it rains at 2pm -- maybe we even have our umbrella on us.
- In contrast -- the unthinkable occurring -- 2+2=5, or a particle having negative mass, is *infinitely surprising!*

If we have two competing models for a phenomenon (eg presidential elections), and one is repeatedly declaring the results to be "very surprising", or "probably due to chance," whereas the other predicts the outcomes well... we likely consider the latter to be more correct.

Put another way, imagine that we're playing a ==prediction game==!
All players are given "Surprise tokens," which they allocate to different outcomes. Any time that the outcome occurs, the player takes on that making points (this reminds us of betting). The goal is to attain the lowest score -- a ==principle of least-surprise!==

We play this "surprise game" for several games, taking on surprise tokens when appropriate. Whoever has fewer tokens -- whoever was *less surprised* is declared the winner.

# Surprise and Probability
- Before we can go any further, we need to quantify the relationship between surprise and probability.
	1. ==Our intuition would tell us that surprise has *some sort* of inverse relationship with probability.== The victory of a team that's *down* by two touchdowns is *more surprising* than the victory of a team down by only a single field goal -- and it's also *less probable*.
	2. We also think that the ==increase in surprise happens *smoothly* as we decrease probability== -- It's surprising to roll a 1 on a 100-sided die, and slightly more surprising to roll a 1 on a 101-sided die.
	3. Lastly, we know that an event is certain to occur has probability 1, and when we want to know the probability that any pair of independent events occur, we multiply their probabilities.
		- i.e., if we look at the probability that {some event x} occurs AND {some other, independent event with probability=1} occurs, this is just equal to the probability of the original event, `p(x)`.
			- `p(x & C) = p(x) * p(C) = p(x) * 1 = p(x)`
		- We can use a similar logic to get a rule for surprise. We know that an event that is certain has a surprise of zero -- ==how can we combine 0 and some number to get that same number back==?
			- ==We add them==! (We can't multiply 0 and x to get x back!)
			- `Surprise(x & C) = S(x) + S(C) = S(x) + 0 = S(x)`

This combination of rules uniquely defines a function that takes probabilities and returns surprises. That function is:

==Surprise Definition==
$S_p(x) = \log( \dfrac{1}{p(x)} )$   

Does the above formula satisfy all of our criteria?
1. Do impossible events give infinite surprise? 
	- Yes, because 1/.0000000... = $\infty$, and $\log(\infty)$ = $\infty$ 
2. Increases in surprise happen smoothly as we decrease probability
	- Yes, $\log(x)$ is a smooth function
3. Independent events give additive surprises 
	- Yes. If there are two events x, y, then $\log(\dfrac{1}{p(x)p(y)})$ , and I believe that `log(ab) = log(a) + log(b)` , so we could break this into addition of surprises.

Note that there's no set base for our logarithm -- surprise doesn't have inherent units.
- If we choose base-2, common in telecommunications, the units are called "bits" or "binary digits"
- If we choose $e$ , common in statistical physics, we get units called "nats", or "natural digits"

# Comparing Surprises

### Comparing with other models
- In our probability game, every time an event occurs, we can take the probability that each competing model assigns to that event and compute the surprise.
- We might be tempted to just say that the least-surprised model is the best one, but we know that one instance/trial isn't enough! We need to take a look at how surprised these models are over repeated instances/samples. We might use an equation like this (average surprise of model) to compare our models:

$\bar{S}_Q = \dfrac{1}{N} \sum{S_q(x)}$ 

Above:
- N gives us the number of repetitions
- Q refers to some particular model
- $S_q$ refers to the surprise that the model Q assigns to the event x, derived from its probability distribution q.

### Comparing with the Truth
- Above, we were implicitly assuming that there is a probability distribution over results -- let's call that distribution $p(x)$. This is "nature's probability distribution".
- With this idea, we can reformulate the relationship above as:

$\bar{S}_Q(P) = \sum p(x) * \log(\dfrac{1}{q(x)})$

Above:
- This is the average surprise of the model $Q$ when its inputs come from the distribution $p$ of some model $P$.

This form of the average surprise has the advantage of providing exact answers, but the disadvantage of requiring an analytical form for $p(x)$, which can be hard to come by.

But if we do have that form, then we can put $P$ in for $Q$ in the expression above and get:

$\bar{S}_Q(P) = \sum p(x) * \log(\dfrac{1}{p(x)})$ 
This is how surprised someone would expect to be, when we have the correct model for the random variable.


# Entropy and Surprise
- Now let's take a step back and compare with standard notation and nomenclature.
- The average surprise of a model Q isn't the usual first step in information theory -- instead, the average surprise of the correct model is the basic entity. It's known as "==entropy==", and its symbol, $H$, is:

==[[Entropy]] Equation==
$H(P) = -\sum{p(x)log(p(x)}$  

Above: 
- Note that the rules of log have been used to turn $log(1/p)$ into $-log(p)$ 
- Why the notation above is standard ***frazzles*** the author, because it emphasizes *log probabilities* rather than the central notion that we call "surprise" (or, often, "==negative log-probability==")

The ==traditional interpretation of entropy is that it corresponds to *the minimum possible average length* of an encoded message==.

This feels ***inherently unsatisfying*** to the author, as a definition for such a basic notion in our understanding of knowledge and inference.

A ==reductive but potent view, courtesy of a purely abstract mathematical approach to probability distributions, interprets entropy as a measure for the *flatness* of a distribution==: A higher entropy is a more-flat distribution. This makes entropy a sort of "summary" of the distribution, just like the mean or the variance.

From an empirical bent:
- The average surprise of some model Q... we started with observations $x$ and models $Q$ claiming to explain *the observations*, rather than the knowledge of the absolute truth implied by $p$.

This average surprise has a name in more traditional approaches: it's the *==cross-entropy==* of Q on P! You might see it written as:

==[[Cross-Entropy]]== formula, used to 
$H(P,Q) = - \sum{p(x)log(q(x))}$ 

From the traditional, Shannon perspective, the interpretation of this quantity is that it is the length of encoded messages using a code optimized for a distribution $Q$ on messages drawn from a distribution $P$ (The author notes that this isn't very intuitive).

The ==[[KL Divergence|Kullback-Leibler Divergence]]== (KL Divergence):

$D_{KL}(P,Q) = H(P,Q) - H(P)$   

Above: The KL Divergence from distribution P to Q (non-symmetrical) is the Cross-Entropy between P and Q minus the Entropy of P.

The KL divergence is a primitive notion of distance that captures some basic amount of the structure of the set of probability distributions, and lets us define "convergence in distribution".

It can be termed as the "excess surprise"; from the Shannon perspective, the KL-divergence is just the "extra bits" when using a sub-optimal coding scheme $Q$ on messages from $P$.

# Surprise with Multiple Variables
- What do we do when we have multiple, possibly related, random variables measured at the same time, or during the same experiment?
	- What does the weather in SF tell us about the weather in Berkeley? In Beijing?
	- We might expect that there's a relationship between SF and Berkeley weather, but not much of one between SF and Beijing.
- At the center of this idea is the idea of statistical independence, or the lack of relationship between two random variables.
- We should be less surprised by the weather in one location if:
	- We know the weather in another location
	- We know the relationship between the weather in these two location
![[Pasted image 20240313164604.png|300]]
![[Pasted image 20240313164555.png|300]]

In general, they are temperate cities, but occasionally things get a little bit hot -- but what do the *joint probabilities* of these look like?
- We can visualize these like a *contour plot*; the joint probability distribution assigns a value to each point in the x-y plane.

![[Pasted image 20240313164724.png|300]]
Above: An example of what mostly-independent temperatures might look like, and what mostly-dependent temperatures might look like (where as the temperature in Berkeley increases, the temperature in San Francisco is also increasing).

# Surprise and Information
- The quantity we describe above as the "excess surprise from using an *independent* model of a *dependent* system" is special enough to get its own name -- ==Mutual Information==.
- The mutual information between two variables, $I(X; Y)$ expresses how much one variable can tell you about another; it serves as a natural measure of statistical dependence -- sort of like correlation, but capable of capturing arbitrarily complex relationships.

# Entropy, Information, and Neuroscience
- Shannon showed that there are optimal ways to encode information so that it can be communicated using as little energy as possible, and as quickly as possible.
- These efficient encoding schemes confer an evolutionary advantage on organisms that use them, so surely we should expect to find neural codes that are optimal from the perspective of information theory!
- The fundamental job of the nervous system is not merely to communicate information efficiently, but rather to construct (from noise-corrupted, ambiguous, and incomplete sensory data) an internal model of the outside world that enables superior action selection for the sake of survival and reproduction.
	- We should expect that the excess surprise about the state of the *outside world* of a model that uses observations of neural data should be minimized.
	- Put another way, if a neuron is representing about the state of the outside world, the mutual information between that neuron's state and that part of the world' state should be high.


# End Matter
In a very real way, no instance is ever repeated, no matter how carefully we control our environment -- something will always escape our grasp (the barometric pressure, the tone in our voice as we ask a survey question, the arrangement of electrons on Jupiter, etc).
- For all of these confounds, we have no *a priori* reason to exclude them, and for some, we know that they have a (potentially small) effect.

...



