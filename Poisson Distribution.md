References:
- Video: [Luis Serrano's The Binomial and Poisson Distributions](https://www.youtube.com/watch?v=sVBOSwT5K8I)

A discrete probability distribution expressing the probability of a given number of events occurring in a fixed interval of time or space, assuming these events occur with a known average rate and independently with respect to the last event.

Imagine that you have a store, and you see that on average, 3 people enter the store.
- Given this information, what's the probability that over the next hour, 5 people enter the store? ((This seems to assume something about the distribution from the average?))
Can be seen as the limit of the [[Binomial Distribution]]

![[Pasted image 20240706105928.png]]

This actually follows from the binomial distribution, using a limit!
Let's start with a simpler problem.

We divide the hour into ten intervals (of 6 minutes each), and assume that only one person can enter/leave any of the 6 minute blocks.
![[Pasted image 20240706111213.png]]
Given an average of 3 people per hour (which we can interpret as P(H)=.3, in our 10-bin case), we can write this as:
- P(5 people show up in an hour) = $0.3^5 * 0.7^5$ 
- There are a $10 \choose 5$ ways to have 5 people arrive, over 10 time bins.
- So the answer $0.3^5 * 0.7^5 * {10 \choose 5}$ = 0.1029193452

What if we did the same thing, but we increased the number of bins in our hour?
![[Pasted image 20240706111330.png]]
= 0.102845784
(See that the reduced "probability" of a bucket is compensated by the increased combination size)

If we keep increasing the number of bins to some large N
![[Pasted image 20240706111715.png]]

As N goes to infinity...
![[Pasted image 20240706111905.png]]
If we plot:
![[Pasted image 20240706111915.png]]

![[Pasted image 20240706112219.png]]

![[Pasted image 20240706112338.png]]
