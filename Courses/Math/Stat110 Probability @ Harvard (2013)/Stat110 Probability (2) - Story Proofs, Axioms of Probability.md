https://www.youtube.com/watch?v=FJd_1H3rZGg&list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo&index=2

---

Recall:
![[Pasted image 20240703190027.png]]

If I have 10 people, and I want to split them into a team of 6 and a team of 4, how many ways are there to do that?
- $10 \choose 4$ 
- Alternatively, $10 \choose 6$
In either case, the remainder is the team of 6 (or 4)
We actually just proved that $10 \choose 4$  = $10 \choose 6$  (because the denominators are 6!4! and 4!6! respectively)

On the other hand, if we wanted to select two teams of 5
- Then if we just did $10 \choose 5$ then we're off!
If we assume teams are 1-5 and 6-10, then there's only one way to do that
In this case, we're not designating some distinction between the two teams (ie there's not a red team and a blue team, there are just two teams)
So in this case it's $10 \choose 5$/2 , because we're double-counting...
There's a clear difference between a team of 4 and a team of 6, but two teams of 5... unless they have different jerseys, it's equivalent!
==The above is a subtle distinction that gets missed==
- It's a little too simplistic to say "order matters or order doesn't matter"; think correctly!


![[Pasted image 20240703190536.png]]
Last time we mentioned that 3/4 of these entries were "obvious" from the multiplication rule; let's talk about the fourth one, the top-right one.

So where does it come from?

----
## Proof Example 
Problem: 
- We want to pick k times from a set of n objects.
- Order doesn't matter, and we're sampling with replacement.

Last time we stated it was $n+k-1 \choose k$ ... but why? Let's get some intuition for it.

Let's check some extreme cases:
- Where k=0 (meaning you don't pick anything): $n-1 \choose 0$ is always 1. (Anything choose zero is always 1; you just don't choose)
- Where k=1 (meaning just pick one): $n \choose 1$ is always n. Note that if we pick once, there's no difference if there's replacement or not.
- Where n=2: $2+k-1 \choose k$ = $k+1 \choose k$ = k+1  (think: If there are 3 objects ABC and we can pick 2 of them, we can pick AB AC BC = 3 = 2+1).

Equivalently: How many ways are there to put k indistinguishable particles into n distinguishable boxes?
n=4 boxes, k=6 particles
This is important for counting... 
![[Pasted image 20240703192521.png]]
All we need to do is convert this picture into something simpler:
We'll make up a little code... with dots and separators: ...||..|.
- Once we have this representation... in this picture there must be k dots, and there must be n-1 separators.
So how many ways are there to do this?
- There are n+k-1 positions here (k + n-1), and in order to specify our code, all we need to do is specify where the dots are (the remaining positions are the positions for the separators!)
$n+k-1 \choose k$ (because we have k dots) or, alternatively, $n+k-1 \choose n-1$ (because we have n-1 separators)

----

A ==Story Proof== is a proof by interpretation, rather than a proof by (eg) algebra or calculus.

Example 1:
$n \choose k$ = $n \choose n-k$ 
- For example "5 choose 2" is the same as "5 choose 3"
- The each work out to 5!/(3!)(2!) or 5!/(2!)(3!), which are equivalent
The story is the interpretation that we're picking k people out of n. ((He doesn't give the story example, fucking idiot))

Example 2 (a handy identity that's harder to provide without a story):
$n {n-1 \choose k-1} = k{n \choose k}$ 
- You can check this by algebra... but that won't help you remember it or understand it.
- The story proof for this would be to imagine that we're going to pick k people out of n, with one of them designated as the president. There are two approaches:
	- Select the k people in the club (n choose k), and then one of those k must be elected president so k(n choose k).
	- First choose the president from the n people, and then once I've chosen the president for my club, I need to select k-1 from the remaining n-1 people.
That's a proof! Completely rigorous, but gives you some interpretation too.
- We're counting the same thing in different ways.

Example 3 (an identity that will be useful in the course, "Vandermonde's Identity"):
$m+n \choose k$ = $\sum_{j=0}^k {m \choose j}{n \choose k-j}$ 
- So this sum collapses to this one [[Binomial Coefficient]] on the left side.
- This doesn't look straightforward at al, does it? It's helpful that the left side is "self-annotating"; let's think about picking k out of m+n -- that's the story. So how does that relate to the sum on the right?
- m+n is also kind of self-annotating too, right? We have an m-sized group and an n-sized group, and we're adding them together. 
- Say m=3 and n=5
![[Pasted image 20240703194510.png|300]]
And we number the people 1...8, left to right
If we need to select k=5 people total from these two groups... maybe we pick 
![[Pasted image 20240703194553.png|300]]
How many ways are there to do this?
- Obviously we need to pick *some number from the m group* and *some number from the n group* such that the total is equal to k. 
- Suppose I picked j=2 from the first, then I *must* pick k-j from the second group.
	- How many ways are there to pick `j from m` and then `k-j from n`? We can just use the multiplication rule: $m \choose j$$n \choose k-j$ 
	- We then just add up these disjoint cases for every j in 0..k (since we can pick up to k from the first group)

For the ==Non-Naive Definition of Probability==, we need the notion of a ==probability space==, which consists of two ingredients S and P:
- S: The ==sample space==, the set of all possible outcomes of an experiment
- P: A function which takes an event $A \subset S$ as an input, and returns $P(A) \in [0,1]$ as output, such that $P(\emptyset)=0$  and $P(S)=1$.
((TLDR we just need a probability distribution over a space of outcomes)) 

Just from this last pretty un-human-friendly definition, I'm going to choose to look at Stanford CS109 "Probability for Computer Scientists", where the students seem to love the professor. This is a good course, I'm just putting it on pause.