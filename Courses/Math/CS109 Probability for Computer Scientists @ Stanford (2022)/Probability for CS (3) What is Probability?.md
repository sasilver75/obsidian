https://www.youtube.com/watch?v=EGgMCE2AgyU&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg&index=4

-----

Recap:
![[Pasted image 20240706121401.png]]
Above: Types of tasks
In each of these cases, we often thought about distinct versus indistinct cases.
![[Pasted image 20240706121436.png]]
The red ones are the ones that you really need to know.

![[Pasted image 20240706122442.png]]
How many calculations are there? (How many pairs of animals are there)
- "It's just choosing 2 animals out of n"
- If we count this way, will we ever end up with Animal A being compared to itself? No
- In $n \choose 2$, will we get (A,B) and (B,A) double counted? No! We care about the items as distinct, but we don't care about the order as we've chosen them.

Another way of thinking about it is a grid
![[Pasted image 20240706122840.png]]
If we took this to more letters, we can imagine that we're really counting how many things are in the upper-right triangle.
![[Pasted image 20240706122903.png]]
*just for n choose 2,* we're counting in this upper right triangle

--------

The ==Sample Space== S is the *set* of all possible outcomes from an experiment.
![[Pasted image 20240706124905.png]]

The ==Event Space== is some subset of S (for example, maybe you care about getting heads)
![[Pasted image 20240706125137.png]]

![[Pasted image 20240706125321.png]]
(Note that a single outcome in a sample space is an ==Outcome==, and an ==Event== is some set of outcomes that we give semantic meaning to!)

So now that we have that...
![[Pasted image 20240706125532.png]]

![[Pasted image 20240706130111.png]]
E^C is an event complement, meaning the probably of an event *not* happening.

![[Pasted image 20240706130215.png]]
This is what they call in STAT110 the [[Naive Definition of Probability]]
- For example we have two dice; what's the probability that the rolled sum=7?
	- We can count the number that add to 7, out of the total
![[Pasted image 20240706130607.png]]


![[Pasted image 20240706131247.png]]
So it's all about how can you interpret your problem as an equally likely sample space.

![[Pasted image 20240706131703.png]]
Our first step for a probability problem is how to to choose to represent our sample space. 
- Ordered or Unordered?
- Distinct or Indistinct

![[Pasted image 20240706132234.png]]

Q: "Why don't we have a choice? Isn't one way the right way?"
All are valid ways of articulating the outcome space, only two will lead to good answers though.

Remember our naive definition of probability, and how we could use it here:
1. Define our sample space
2. Determine the size of the sample space
3. Determine the size of the event space (the subset of sample space we care about)
4. E/S
This only works if our sample space has all equally-likely outcomes

==4 cows, 3 pigs==
==P(1 cow, 2 pigs) = ?==

It's easier to show that outcomes are *not* equally likely, than it is to prove things as being equally likely.

In the top-right example (indistinct, unordered)
{3 cows} vs {3 pigs} ... are these equally likely? NO! Because we have different proportions of cows and pigs. So we can't use this sample space if we want to use our naive definition of probability!

In the bottom-right example (indistinct, ordered)
$[cow,cow,cow]$ vs $[pig,pig,pig]$ are similarly not equally likely. NO!


In the top-left example (distinct, unordered):
- We have 7 toys, and we want to figure out how many subsets of 3 we can make?

event space/sample space = $?/ {7 \choose 3}$

How many outcomes are there in the event space, for this unordered problem where outcomes look like $\{C_1,P_2,P_3\}$?
- Take my 4 cows, choose one cow
- Take my 3 pigs, choose 2 of them
Counting by steps: The number of outcomes in the first step doesn't effect the number of outcomes in the second. So our two steps look like:
- ${4 \choose 1} * {3 \choose 2}$

So $({4 \choose 1} * {3 \choose 2})/{7 \choose 3}$ = 12/35 = ==.343==

In the bottom-left example (distinct, ordered):
- Outcomes look like $[C_1,P_2,P_3]$
- Sample space looks like us picking 3 ordered items: 7 x 6 x 5 = 210
	- Pick one for first
	- Pick one for second
	- Pick one for third
- Event space: We can pick the cow as either 1st, 2nd, or 3rd item in the sequence
	- If the cow is first: 4 choices for cow x 3 choices for first pig x 2 choices for second pig
	- If second: 3 x 4 x 2
	- If third: 3 x 2 x 4
	- Sum: 72 (Because we're saying the outcomes can come from these mutually exclusive cases, we know they're mutually exclusive).

72/210 = ==.343==


---

![[Pasted image 20240706134349.png]]
The suits can be anything you want, but the values must be inordered.
- Cards are already distinct...

We know that there are $52 \choose 5$ possible hands
How can we count the subset that are straights?

Cards are distinct, ordered, in my head...
The one thing to note is that there JKQ10=16 cards with the value 10, and that an ace can be either a 1 or an 11 (meaning there are A1=8 cards with the value 1, and 4 cards with the value 11)

| Value | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | 11  |     |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Count | 8   | 4   | 4   | 4   | 4   | 4   | 4   | 4   | 4   | 16  | 4   | 60  |
So the question is what's P(Straight), where a Straight is 5 consecutively-ranked cards of any suit?

How can we get started?
- Let's say pick one *value* to get started at, representing the left edge of our straight.

Given that the straight needs to be over 5 consecutive values, this means we could choose any "starting number" in the range \[1...7\]
- (He doesn't really understand straights, I think. He thinks there are 10 possible starting positions, we'll roll with that. Oh, apparently you can actually do 10 jack queen king ace, lol -- so 10 is good.)

Note that 2,3,4,5,6 and 2,4,3,5,6 are both the same straight, so we don't care about order.
So is it just 10/(52 choose 5)?
No! We've just been talking about values, not suits! We haven't sid that it's a 2 of hearts, 3 of diamonds, etc.

So given 2 3 4 5 6
How can I choose a suit for the first case? (4 choose 1) = 4
How about for the next card? ( choose 1) = 4
...

So it's $(10 * 4 * 4 * 4 * 4 * 4)/{52 \choose 5}$

![[Pasted image 20240706141015.png]]

----

![[Pasted image 20240706141249.png]]

I think the way to think about this one is by inverting the problem and asking "What's the probability that NO defects occurred?" Because the complement to that is "at least one defect occurred".
The empirical defect rate seems to be 1/n; this is the probability of a defect of a given chip.
So the probability of no defect for a given chip is 1- (1/n)
So the probability of no defects in a selection of k chips is (1-1/n)^k
So the probability of at least one defect is the complement, 1-(1-1/n)^k
Unsure if this is right
= 0.3702623906705538 Fuck

His walkthrough:
- Let's set some concrete numbers; n=7 chips, k=3 selected for testing
If we can find a way to ==construct the sample space such that outcomes are equally likely==, we can just use counting to attack the problem!

${7 \choose 3}$ is the number of ways of choosing k chips from n. This is our denominating sample space.

But what's our event space, where every outcome has the defective chip in it?
- ${6 \choose 2}$ ways to choose two of the non defective chips
- $1 \choose 1$ ways to choose one defective chip of the 1 defective chips (1)

So it's just ${6 \choose 2}$ /${7 \choose 3}$  = .428














