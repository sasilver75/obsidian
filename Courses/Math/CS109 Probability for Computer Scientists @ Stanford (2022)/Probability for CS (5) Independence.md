https://www.youtube.com/watch?v=zTJDZ2wmaRU&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg&index=6

------

Review: Conditional Probability
P(A,B) vs P(A|B)
[[Chain Rule of Probability]]: P(A,B) = P(A|B)P(B) or P(B|A)P(A)
[[Law of Total Probability]]: P(E) = P(E|F)P(F) + P(E|F^c)P(F^c) (where c is "complement")
[[Bayes Theorem]]: P(A|B)=P(B|A)P(A) / P(B)

---

Let's talk about Gambling in Poker!
![[Pasted image 20240708135936.png]]

![[Pasted image 20240708140051.png]]

We'll actually use this:
![[Pasted image 20240708140212.png]]
52 cards in the deck
We have 7 cards, no aces
T = Tell and A = Opponent has an Ace
P(T|A) = .5
P(T|A^c) = .1

We want to play if the P(A^c|T^c) > .5

We can write this as 
$P(A^c|T^c) = P(T^c|A^c)P(A^c) / P(T^c)$
or, after expanding denominator
$= P(T^c|A^c)P(A^c) / P(T^c|A^c)P(A^c) + P(T^c|A)P(A)$

Given
$P(T|A) = .5$
$P(T|A^c) = .1$

We can infer, via complements
$P(T^c|A) = .5$
$P(T^c|A^c)$ = .9

From our hand of 7 cards (leaving 45) that has no aces (leaving 4), we can infer $P(A)$ using equally-likely outcome spaces:
$P(A^c) = {41 \choose 2}/{45 \choose 2} = 0.8282828283$ = 82.8% chance they *don't* have an ace
- (Number of hands from our deck not including aces / number of hands from our deck including aces)
$P(A) = 1-.082828 = 0.1717171717$ = 17.2%

Now we can plus them all in to our 
$P(A^c|T^c) = P(T^c|A^c)P(A^c) / P(T^c|A^c)P(A^c) + P(T^c|A)P(A)$
$= .9*.8282 / (.9*.8282 + .5*.1717)$
$P(A^c|T^c) = 0.8967193196$
The probability that they don't have an ace, given that they don't have a tell is ~90%
So we should play this game!

Okay, that was the review from last time.

----

# Independence

![[Pasted image 20240708144608.png]]
When it's impossible for both events to occur at the same time.

![[Pasted image 20240708144707.png]]
When it's possible for the events to occur simultaneously
 We just subtract off the middle overlap to not double-coint

![[Pasted image 20240708145022.png]]
![[Pasted image 20240708145030.png]]
OR of three events: All sets of ones, minus all pairs of sets, plus the intersection of the sets


![[Pasted image 20240708145358.png]]
If you can identify that two events are independent, if you want the probability of "AND", you just multiply :)
If they're not independent, you have to revert back to the chain rule.

![[Pasted image 20240708145459.png]]
![[Pasted image 20240708145614.png]]
Above: Backing out the idea that if A and B are indepdnent, the probability of A AND B is just the product of their marginal probabilities.

Independence often makes math much easier to do; in reality, we often assume that some events are independent just because it makes the math easier.


![[Pasted image 20240708145847.png]]



![[Pasted image 20240708150736.png]]
![[Pasted image 20240708150742.png]]
More restating on Independence: When you collapse into the world where B happens A is just as likely as it was without B happening.
![[Pasted image 20240708150808.png]]
Dependence: When you collapse into the world where B has happens, A become relatively more likely.

![[Pasted image 20240708150849.png]]
If two events are independent, then they are also independent of eachothers' complement.

![[Pasted image 20240708151127.png]]
If you have many events and they're all independent, and we want to get the joint probability, we can just multiply the probabilities together.
- Example: Coin flips, which we assume are conditionally independent (even if we're using multiple coins). 

![[Pasted image 20240708151435.png]]
Some examples of determining conditional independence, intuitively


---

Some practice!
![[Pasted image 20240708152009.png]]
Both pairs have one dominant, and one recessive gene
Each parent will pass on one of their genes (each being equally likely) to a child
The child has curly hair only if it has a recessive from both
Assuming independent, the probability of any child having (a,a) is 0.25

But now we have 3 kids, and every kid has a .25% chance of having curly hair.
What's he probability that all 3 have curly hair? .25^3
If we *didn't know independence*, we'd have to use the generalized chain rule in the bottom-right.
Q: Wit the expansion of the chain rule, does the order matter?
A: No!


![[Pasted image 20240708152418.png]]
These are not mutually exclusive, and they're conditionally independent.
The probability that SOME function path exists (meaning at least one is working)
Is the same as 1 - probability that NONE are working
Which is 
$P(E) = 1 - \prod_{i=1}^i{1-p_i}$
	  = complement of {they're all broken}



