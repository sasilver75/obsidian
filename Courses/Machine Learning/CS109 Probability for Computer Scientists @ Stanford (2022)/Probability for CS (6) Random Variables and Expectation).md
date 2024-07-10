https://www.youtube.com/watch?v=8QCg2ur-3fo&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg&index=7

-------

Recap
![[Pasted image 20240709162524.png]]
Some definitional understandings
![[Pasted image 20240709162536.png]]
Relations between probabilities
![[Pasted image 20240709162551.png]]
Understanding probability or OR (in case of mutual exclusion: Add, in case of mutual inclusion, we do A+B-AandB) and in AND (in case of exclusive, multiply, in case of not exclusive use chain rule)

![[Pasted image 20240709162836.png]]
.6^2 + .4 ^2 = 0.52
P((H1 and H2) or (H1^c and H2^c))
These are mutually exclusive events (H1, H2), meaning we can just sum these two probabilities of 
P(H1 and H2) + P(H1c + H2c)
H1 and H2 are independent, so P(H1 and H2) becomes P(H1)P(H2)
.6*.6 + .4*,4 = .52

----

![[Pasted image 20240709163837.png]]
Genes and whether a Trait was expressed
![[Pasted image 20240709163854.png]]
We look for independence!
![[Pasted image 20240709163905.png]]
Here's how we did it: We take every gene, and ask whether it's independent of the trait.
- Count joint probability
- Count marginal probabilities
If two things are independent, it should be the case that the probability of each event multiplied together should be the same as the probability of the "AND" (think: coin tosses).
It seems like the trait might be independent of Gene 4; it seems like knowing about these genes doesn't chance our probability of the trait being expressed.
- Note that just because things are dependent doesn't mean there's a causal relation between the two.

![[Pasted image 20240709165130.png]]
In a world where I tell you about G2, we're now in a world where G5 inisn't 

![[Pasted image 20240709165150.png]]
G5 makes it more likely for G2 to be exprsesed, and G2 makes it more likely for the trait to be expressed!

![[Pasted image 20240709165332.png]]
The bottom is just like how we usually talk about independence as being P(E|F) = P(E).

Q: If two things are conditionally independent, does it tell you anything about whether they're independent hwen *not* conditioning on G?
A: ==A GREAT QUESTION!== The answer is no! Things can be independent, and then, when conditioned on G, be dependent -- or vice versa! Because causality is such a crazy thing!

![[Pasted image 20240709170526.png]]
Independence relationships can change with conditioning!


![[Pasted image 20240709170601.png]]
An example from earlier in the course

![[Pasted image 20240709170617.png]]
We can use the definition of conditional probability P(E|F) = P(EF)/P(F)

But what if we conditioned on more than one movie? What if there were 3 or 4 movies we were conditioning on?
![[Pasted image 20240709170700.png]]
Do you think E4 is independent of E1/E2/E3? Probably not! Knowing about whether you've watched E1/E2/E3 will influence our belief that someone has watched E4

Thus we wan't say that this equals P(E4)

We can use our definition of conditional probability
![[Pasted image 20240709170815.png]]
A problem comes up though! The problem is the numerator...

![[Pasted image 20240709170838.png]]
Let's say we wanted to condition on someone watching 30 movies
If there are 13,000 titles on Netflix, and a user watches 30 random titles.
If E is the event we care about
- Everyone watches 30
- How many watch the exact 4 we care about

Let's try to get an equally-likely sample space to solve the problem.


${13000-4 \choose 26}{4 \choose 4}/{13000 \choose 30}$

Yep! :) 

![[Pasted image 20240709172146.png]]
We can easily get into a situation where we have to compute a probability that's really rare, when working at netflix scale.
They assume that there's another event... and that event is "a user liking foreign emotional comedies," and they say that this event is what really governs whether a user likes any of these four movies
- They then make the wrong, but HELPFUL assumption, saying that the probability that you like any of these movies are conditionally independent of eachother, given that we know that you like foreign emotional comedies.
![[Pasted image 20240709172254.png]]
Then everything is just equal to P(E4|K1)!
((But... you can have a bunch of other Ks too, like "they like action", and then you have another layer of variables?))


