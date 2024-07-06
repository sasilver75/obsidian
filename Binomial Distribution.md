References:
- Video: [Luis Serrano's The Binomial and Poisson Distributions](https://youtu.be/sVBOSwT5K8I?si=11EhVauH1eA_zMx0)

Say we have a coin with P(H)=0.3
And we flip it 10 times and get:
![[Pasted image 20240706102214.png]]
Resulting in 4 heads, which isn't exactly what we expect.
If we run this experiment hundreds and thousands of times, the average number of heads observed will converge to 3 heads.

So given that coin, what's the probability of getting X heads?
![[Pasted image 20240706102409.png|200]]
Note that these probabilities add to one. 
![[Pasted image 20240706102513.png]]
The two parameters of the binomial distribution are
- The number of throws
- Probability of heads

![[Pasted image 20240706103131.png]]

Using our formula from [[Probability for CS (2) Combinatorics]], we can us $n!/k!(n-k)!$
$10!/(2!*8!)$ = 45 ways in which we could get two heads
![[Pasted image 20240706103827.png]]
(It can also be thought of as 10 * 9 / 2; since there are 10 ways to choose the first coin, and 9 ways to choose the second, and then we're dividing by 2 to account for overcounting)

![[Pasted image 20240706104510.png]]

![[Pasted image 20240706104558.png]]




