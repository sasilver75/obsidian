#article 
Link: https://colah.github.io/posts/2015-09-Visual-Information/

--------

Information theory gives us precise language for describing a lot of things. 
- How uncertain am I?
- How much does knowing the answer to question A tell be the answer to question B?
- How similar is one set of beliefs to another?

Information theory crystallizes vague theories about these into precise, powerful ideas.
Unfortunately, information theory can be a little intimidating, but it doesn't have to be!

# Visualizing Probability Distributions
- Before we dive into information theory, let's talk about how we can visualize simple probability distributions.
- We can visualize a single probability distribution: 
![[Pasted image 20240313170239.png|50]]
- What if we want to visualize two probability distributions at the same time? It's easy if they *don't interact* (if they're statistically independent) -- we can just use one axis for one variable and one for the other!

![[Pasted image 20240313170332.png|200]]

- Notice that straight and vertical lines that go all the way through -- that's what *independence* looks like! 
	- The probability that we're wearing a coat *doesn't change* in response to the fact that it'll be raining next week!

- When variables interact, there's *extra* probability for particular pairs of variables, and *missing/reduced* probability for others!
	- If we consider {weather today} and {clothing today}, then there's "extra probability mass" that I'm wearing a coat and it's raining, because the variables are correlated, making eachother more likely. It's more likely that I'm wearing a coat on a day that it rains than the probability of coat+sunshine.

![[Pasted image 20240313170727.png|300]]

While this might *look* cool, it isn't very useful for understanding what's actually going on.

Instead, let's focus on one variable like the weather!
- We know how probable it is that it's sunny or raining.
	- For both cases, we can look at the *conditional probabilities* -- how likely am I to wear a T-shirt *if it's sunny?* How like am I to wear a coat *if it's raining?*

![[Pasted image 20240313171240.png]]

There's a 25% chance that it's raining. *If it's raining*, there's then a 75% chance that I'd wear a coat! So the probability that it's raining *and I'm wearing a coat* is 25% *times* 75%, which is approximately 19%.

We can write this as:

$p(rain, coat) = p(rain) * p(coat | rain)$ 

This is a single case of one of the most fundamental ideas of probability theory!

$p(x,y) = p(x) * p(y|x)$      (which can also be written as)   $p(x,y) = p(y)*p(x|y)$ 

Above, we're *factoring* the distribution, breaking it down into the product of two pieces.
- First we look at the probability of one variable, and then we look at the probability of the other variable. The choice of which variable to start with is arbitrary.

![[Pasted image 20240313212905.png|450]]

Note that the labels have different meanings than the previous diagram. Here, the t-shirt and coat are now *==marginal probabilities==* -- the probability of me wearing that clothing without consideration of the weather.
- On contrast, the probabilities in the table are *==conditional probabilities==*

# Codes 
- Now that we can visualize probability, let's dive into information theory! 

Situation:
> Bob is our friend who likes animals. He likes them so much that he only ever says four words: "dog", "cat", "fish", and "bird."
> A few weeks ago, Bob moved to Australia and decided that he would only communicate with us in Binary! All of our imaginary messages from Bob look like this:
> "1 0 0 1 1 0"
> To communicate, Bob and I have to establish a code, a way of mapping words into sequences of bits.
![[Pasted image 20240313220741.png|200]]
> To send a message, Bob replaces each symbol with the corresponding keyword, and then concatenates them together to form the encoded string.
![[Pasted image 20240313220834.png|200]]


# Variable-Length Codes
- Unfortunately, communication services in imaginary-Australia are expensive! We have to pay $5 per bit of every message we receive from Bob, and Bob likes to talk a lot!
- We have to investigate: Is there any way that Bob and I can make our average message length even *shorter?*
- It turns out that *Bob doesn't say all of the words equally often!*
	- Bob really likes dogs! He talk about them all the time -- mostly about Dogs though!
![[Pasted image 20240313221157.png|200]]

So our old code uses codewords that were 2 bits long, regardless of how common they are!

![[Pasted image 20240313221305.png|350]]
Above: 
- We use the vertical axis to visualize the probability of each word, and the horizontal axis to visualize the *length* of the corresponding codeword, L(x).
- ==We can see that the (probability-weighted) average length is 2 bits per word!==

Perhaps we could be very clever and make a variable-length code where codewords for common words are made especially short?

![[Pasted image 20240313221552.png|300]]
(You might wonder, why not use 1 by itself in a codeword, above? Sadly this would cause ambiguity when we decode encoded strings -- we'll talk more about this later).

Let's visualize this encoding again:

![[Pasted image 20240313221644.png]]
Above:
- If you actually multiply it out, you can see that the ==probability-weighted average length==  here is only 1.75!
	- $(.5 * 1) + (.25 * 2) + 2(.125 * 3) = .5 + .5 + .75 = 1.75$

It turns out that this code is the best possible code. There is no code which, for this distribution, we'll get an average codeword length of less than 1.75 bits.

There is simply a fundamental limit -- we call this ==optimal average length of encoding== the fundamental limit, or the [[Entropy]] of the distribution!
- It seems that the crux of the matter is understanding the tradeoff between making some words short, and other words long! Once we understand that, we'll be able to understand what the best possible codes are like.

# The Space of Codewords
- There are *two codes* with a length of 1 bit: 0 and 1.
- There are *four codes* with a length of 2 bits: 00, 01, 10, 11
- There are *eight* codes with a length of 3 bits: 000, 001, 010, 011, 100, 101, 110, 111

Although we can *represent* 8 sequences with 3 bits, we might have complicated mixtures of *codewords* like two codewords of length 2, and four codewords of length 3.
What decides how many *codewords* we can have at different lengths?

Recall that Bob was encoding his messages into strings by concatenating them:

![[Pasted image 20240313223443.png|250]]

There's a slightly subtle issue to be careful of, when crafting a variable length code: ==How do we split the encoded string *back* into codewords in a deterministic way?==

We want our code to be *==uniquely decodable==, with only one way to decode an encoded string!*
- One way to do this would be to have some sort of "end of codeword" symbol, but we don't -- we're only sending 0s and 1. 
- We need to be able to look at a sequence of concatenated codewords and tell where each one stops
- It's very easy to make codes that *aren't* uniquely decodable. For example, imagine that 0 and 01 are both codewords -- it would be unclear what the first codeword of the encoded string 0100111 is -- it could be either.
	- The property that we want is that if we see a particular codeword, there shouldn't be a *longer* version that is *also* a codeword. In other words, ==no codeword should be the prefix of another codeword!==
		- This is called the *==Prefix Property==* and codes that obey it are called prefix codes.

One way to think about this is that every codeword requires a sacrifice from the space of possible codewords.

If we take the codeword 01, we lose the ability to use any codewords that it's a prefix of:
- We can't use 010 
- We can't use 011010110
They're lost to us!

![[Pasted image 20240313224323.png|300]]
Above:
- Since a *quarter* of all possible codewords start with 01, we've sacrificed a quarter of all possible codewords!  
- This is just the price we pay in exchange for having one codeword that's only 2 bits long! In turn, this sacrifice means that all of the other codewords need to be a bit longer. But what's the right amount/nature of tradeoff?

# Optimal Encodings
- You can think of this like having a limited budget to spend on getting short codewords.
	- We pay for one codeword by sacrificing a fraction of possible codewords!
- What's the cost of buying various codewords?
	- The cost of buying a codeword of *length 0* - is 1 -- all possible codewords. It turns out if you want to have a codeword of length 0, you can't have any other codewords.
	- The cost of buying a codeword of *length 1* is is 1/2, because half of all codewords start with (eg) 0.
	- The cost of a length 2 codeword (like "01") is 1/4, because a quarter of all possible codewords start with 01.
	- In general, the cost of codewords decreases exponentially with the length of the codeword.

![[Pasted image 20240313232651.png]]

We want short codewords because we want short average message lengths! Each codeword makes the average message length *longer* by its probability times the length of the codeword! (Think: the area of the squares in the preceding diagrams).

![[Pasted image 20240313232818.png|200]]

These two values are related by the *length* of the codeword:
- The amount we pay decides the length of the codeword
- The length of the codeword controls how much it adds to the average message length.

We can picture them together:
![[Pasted image 20240313232855.png]]

==Short codewords reduce the average message length but are expensive, while long codewords increase the average message length but are cheap.==

![[Pasted image 20240313232925.png]]
What's the best way to use our limited budget?
1. We want to spend more of frequently-used codewords, so let's distribute our budget in proportion to how common an event is!
	- So if one event happens 50% of the time, we should spend 50% of our budget buying a short codeword for it -- and if an event only happens 1% of the time, we should only spend 1% of it.
	- It turns out that this is both the "natural" thing to do, but it's also the optimal thing to do! (Skipping proof)


# Calculating Entropy

Recall that the *cost* of a message of length $L$ is $1/2^L$ 
- We can invert this to get the length of a message that costs a given amount: $log_2(1/cost)$ 
- ==Since we spend $p(x)$ on a codeword for $x$, the length is $log_2(1/p(x))$== 

![[Pasted image 20240321221128.png]]
Earlier, we discussed that there's a *fundamental limit* to how short one can get the average message to communicate events from a particular probability distribution, p.

This limit -- ==the average message length using the best possible code== is called the [[Entropy]] of p, H(p)

Now that we know the optimal lengths of the codewords, we can actually calculate it:

$H(p) = \sum_x(p(x)log_2(1/p(x)))$ 

Using the identity of $log(1/a) = -log(a)$    ... we can rewrite the above as the more common:

==Entropy Equation==

$H(p) = -\sum(p(x)log_2(p(x)))$ 

==No matter what I do, on average we need to send *at least* that number of bits if I want to communicate which event occurred.====

The average amount of information needed to communicate something has clear implications for ==compression==... but it also describes how ==uncertain== we are.
- If I knew for sure what was going to happen, we wouldn't have to send a message at all! If there's *two* things that could happen with 50% probability, I only need to send 1 bit.
- If there 64 things that could happen with *equal probability*, we'd have to send 6 bits (2^6=64)
	- ==The more concentrated the probability, to more we can craft a clever code with short average messages!==
	- ==The more diffuse (uniform) the probability, the longer our messages have to be.==

==The more uncertain the outcome, the more we learn when we find out what actually happened.==

# Cross-Entropy

> Let's say that Bob married Alice.
> Alice wasn't a dog lover like bob -- she was a *cat* lover!

![[Pasted image 20240321223329.png|300]]

The two of them say the same words, but at different frequencies.
If Alice uses the same encoding scheme that we devised for Bob above, her messages are going to be longer than they need to be. 
Bob's code is optimized to his probability distribution, but Alice has her own probability distribution.

Though the average length of a codeword when *Bob* uses his own code is 1.75 bits, when Alice re-uses his code for *her* messages, the average length is 2.25 bits!

This difference in these two lengths is called the [[Cross-Entropy]]
==This extra length (2.25-1.75) -- the average length of communicating an event from *one* distribution using the optimal code from *another* distribution -- is called the Cross Entropy!==

Formally:

$H_p(q) = \sum_xq(x)log_2(\dfrac{1}{p(x)})$ 
above: 
- Note the use of `q` and `p` to represent two *different* distributions!
- Note that Cross Entropy is *not symmetrical!*
- ==Here, it's the cross-entropy of Alice the cat lover's word frequency with respect to Bob the dog-lover's word frequency.==

![[Pasted image 20240321225205.png]]

To keep the cost of communications down, let's ask Alice to use *her own code!*
- After she did this, Bob sometimes accidentally used Alice's code -- ==surprisingly, it's worse for Bob to accidentally use Alice's code than for Alice to use his!== This is because ==Cross Entropy is not symmetric!== 

#### Why care about Cross-Entropy?

[[Cross-Entropy]] gives us a way to express how different two probability distributions are.
- The more *different* the distributions of p and q are, the more the {cross-entropy of p with respect to q} will be bigger than the entropy of p.
![[Pasted image 20240321233454.png|250]]

The interesting thing is the ==difference between the entropy and the cross-entropy==.
- This difference is how much longer our messages are because we used a code optimized for a different distribution -- ==in another word, the "excess" length by using the "wrong" encoding for our messages==.
	- This difference is called the [[Kullback-Leibler Divergence|Kullback-Leibler Divergence]], or ==KL Divergence==.

The KL Divergence of *p* with respect to *q* is defined as:
$D_q(p) = H_q(p) - H(p)$
KL divergence of p with respect to q is the Cross entropy of p with respect to q, minus the entropy of p.

The neat thing about the KL divergence is that ==it's like a distance between two distributions==
- ((But careful with this idea of distance, because KL Divergence, like Cross Entropy, isn't symmetrical))

==Cross Entropy and KL Divergence are incredibly useful in ML -- oftentimes we want one distribution to be close to another==! For example, we might want a predicted distribution to be close to the ground truth distribution.


# Mutual Information
- We observed that knowing one variable can mean that communicating another variable requires less information -- one way to think about this is to imagine amounts of information as bars.
- ![[Pasted image 20240322000721.png]]
- These bars overlap if there's shared information between them.
- For example, some of the information in X and Y is shared between them, so H(X) and H(Y) are overlapping bars, and since H(X,Y) is the information in both, it's the union of the bars H(X) and H(Y).

Once we think about things this way, a lot of things become easier to see!
- We previously noted that it takes more information to communicate both X and Y (the "==joint entropy==", H(X,Y)) than it takes to *just* community X (the "==marginal entropy==", H(X)) 
- But if you already KNOW Y, then it takes *less information* to communicate X! (the "==conditional entropy==", H(X|Y)) than it would if you didn't know Y!

![[Pasted image 20240322001020.png]]

==H(X|Y) is the information we need to send to communicate X to someone who already knows Y -- it's the information in X which isn't also/already in Y!==
- Visually, this means that H(X|Y) is the part of H(X) bar that *doesn't* overlap with H(Y)

![[Pasted image 20240322004331.png]]

Another identity: H(X,Y) = H(Y) + H(X|Y)


This information shared between variables -- the intersection of their information, is called [[Mutual Information]], and is defined as:

==Mutual Information Equation==

$I(X,Y) = H(X) + H(Y) - H(X,Y)$

This definition works because H(X) + H(Y) has two copies of the mutual information, since it's in both X and Y, whereas H(X,Y) only has one.

In contrast, the ==Variation of Information== is information that *isn't shared* between the two variables!

$V(X,Y) = H(X, Y) - I(X, Y)$

Variation of Information gives us a metric, a notion of distance, between different variables -- the variation of information between two variables is zero if knowing the value of one tells you the value of the other. As variables get more independent, the variation of information increase.

How does this relate to KL divergence?
- KL divergence gives us a distance between two distributions over the same variable or set of variables.
- In contrast, variation of information gives us distance between two jointly distributed variables.
- ==KL divergence is *between distributions* and Variation of Information is *within* a distribution.==

![[Pasted image 20240322004936.png]]

# Conclusion

- If we care about communicating in a minimum number of bits, these ideas are clearly fundamental.
- But people in ML, thermodynamics, genetics, and physics typically don't care about information theory because they want to compress information -- they care because it has a compelling connection to their field!
- Information theory turns up in all places because it offers concrete, principled formalizations for many things we need to express. It gives us ways of measuring and expressing uncertainty, how different two sets of beliefs are, and how much an answer to one question tells us about others: how diffuse probability is, the distance between probability distributions, and how dependent two variables are.

In Machine Learning
- Let's say we're classifying a picture as either a dog or cat.
- Let's say we predict the answer is 80% dog, and 20% cat -- how much better would it have been to say 85%? The correct answer depends on what we're using the model for
	- Do we only care about whether the top guess was correct (accuracy)
	- Do we care about how confident we are in the chosen answer?
		- How bad is it to be confidently wrong?

Information gives us a powerful new framework for thinking about the world -- sometimes it perfectly fits the problem at hand, and other times it's not an exact fit, but is still extremely useful.


