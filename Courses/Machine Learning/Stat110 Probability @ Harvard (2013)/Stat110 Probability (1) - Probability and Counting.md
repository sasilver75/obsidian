https://www.youtube.com/watch?v=KbB0FjPg0mw&list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo

A lot of this course is about just getting better at pattern recognition, and knowing how to apply the tricks in your toolbox. It takes practice to get good at probability!

---

A ==sample space== is the set of all possible outcomes of some random experiment
- Interpret the word "experiment" in an extremely broad manner.

An ==event== is a *subset* of the *sample space*.

One of the things that make probability difficult is that we're going to be doing things that are deeply counterintuitive to almost everyone.

The ==Naive Definition of Probability== (which can only be used when you have strong justification for doing so) is that the probability of an event A, P(A) is equal to (# of possible favorable outcomes / # of possible outcomes).
- So if we flip a coin twice, there are four possible outcomes. The sample space is $\{(H,H), (H,T), (T,H), (T,T)\}$ ... and so we have a 1/4 chance of $(H,H)$. We didn't say if it was a "fair" coin or not.
	- This is an example where it's straightforward to count the number of possible outcomes; the size of the sample space. But for many problems this isn't easy to do!
- This definition assumes that all outcomes are equally likely (a very strong assumption in many problems), and that we have some finite sample space (with which to use as our denominator).


Counting
- ==Multiplication Rule==
	- If we have an experiment with $n$ possible outcomes, and then for each outcome we do a second experiment with $m$ possible outcomes, then in total there are $nm$ outcomes.
	- Example:
		- Flipping two coins in a row results in 2x2=4 possible outcomes.
		- If you have two pairs of shirts and 3 pairs of pants, you have 6 possible outfits, and it doesn't matter whether you choose your shirt or pant first.


[[Binomial Coefficient]] (pronounced n choose k)
- Say we have n people; how many ways can we choose k of the n people, where order doesn't matter.
![[Pasted image 20240628185546.png]]
Also 0, if k > n

This equation actually follows from the highlighted multiplication rule above
Say we have n people to select from, and we want to choose k
- First: We select from n
- Second: We select from n-1
- ...
- Last: We select from (n-k+1)
But we could have selected these people above in *any order*, so we have to divide by k-factorial!

This is the same thing as:
![[Pasted image 20240628190210.png]]
Once you cancel some terms out.


==Example: So when considering the probability of a full-house (3 of one rank, 2 of another)==
- We can formulate this probability as (some number of acceptable hands / some number of total hands)
- We know ${52 \choose 5}$ is the number of order-agnostic hands of cards we could draw, so that will be our denominator

For the numerator, we could imagine drawing a tree out:
- At the first branching, choosing some rank (say, 7, but it could be any rank, of which there are 13), and we need 3 out of the 4 cards of that rank.
- So this is $13 * {4 \choose 3}$ 
- And then we need some other rank (say, 4, but it could be any rank, of which there are 12 remaining), and we need 2 out of the 4 cards of that rank.
- So this is $12 * {4 \choose 2}$

![[Pasted image 20240628191005.png|300]]
So this is the final solution! He recommends visualizing this tree structure, and the multiplication rule.

S "n choose k" is an example of binomial coefficient, where we choose n from k, where order doesn't matter.

**But what if order *does* matter?**

What we're doing here is sampling, so let's talk about a sampling table.

This is going to be a 2x2 table, and we'll try to fill it in!
- We have some population of items, or people, or anything, and we're drawing a sample. We choose k objects out of n, and want to know how many ways there are to do it.

He calls this a *sampling table*:

|                                                                             | Order matters                       | Order doesn't matter |
| --------------------------------------------------------------------------- | ----------------------------------- | -------------------- |
| Sampling with Replacement<br>(pick one, put it back, pick one, but it back) | n^k<br>from the multiplication rule | ${n+k-1 \choose k}$  |
| Sampling without Replacement                                                | n(n-1)...(n-k+1)                    | ${n \choose k}$      |
|                                                                             |                                     |                      |
We don't want to memorize the table above, we want to understand how each of these follow from the multiplication rule.
- The top-right one is the only one that's much more subtle, but still useful.
 




