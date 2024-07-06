https://www.youtube.com/watch?v=ag4Ei15CG0c&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg&index=2

[[Permutation]]s and [[Combination]]s

----

Recap:
![[Pasted image 20240705170331.png]]
Counting with Steps
- If we have a counting problem with many parts, where the first part can result in m outcomes, and doesn't constrain the number of outcomes of the second experiment, then we can just multiply the number of outcomes together `m*n`
Counting with Or
- If the outcome of an experiment can be drawn from either Set A or Set B, then the number of outcomes of the experiments is `|A or B| = |A| + |B| - |A and B|` (subtracting off the outcomes that are in both groups so as not to double-count)

![[Pasted image 20240705170516.png|300]]
This is a good problem to understand both ideas above
- We count the number of outcomes in each set A and B, and then remove from our count the number the overlapping items.

----

How many letter orderings are possible for the following strings?

CHRIS

![[Pasted image 20240705170849.png]]

Choose the first letter (5 choices), choose the second letter (4 choices), choose the third letter (3 choices), etc.
- 5 factorial = 120
This is the *unique* number of orderings.

![[Pasted image 20240705174434.png]]
You're the FBI and you've found a phone, and it's got smudges on exactly 6 of the numbers... but you don't know what the ordering is. How many unique 6 digit passcodes are possible?
(Assuming that all of the numbers have to be used)
Choose the first (6), choose the second (5), ...
`6!`

What if we said: Forget the smudges, and passcodes have 6 digits in them, and each of the numbers have to be different?
10 choices for first
9 choices for second
8 choices for third
7 choices for fourth
6 choices for fifth
5 choices for sixth

10 x 9 x 8 x 7 x 6 x 5

Can be rewritten as `(10! / 4!)`

![[Pasted image 20240705174800.png]]
We've talked about how to order objects if they're distinct...
- But this gets more interesting if we let some objects get INDISTINCT/indistinguishable from one another

Is $10_a10_b0_c$ different from $10_b10_a0_c$ ?

![[Pasted image 20240705174909.png]]
This is the exact same problem; how many ways are there of ordering this cans, assuming the same-classes cans are indistinguishable?

Pro tip: Even though they're indistinguishable, let's *make them* be distinguishable!

![[Pasted image 20240705175048.png]]
We know that there are 5! ways of ordering the cans if they were distinct.
But in this example, we're overcounting the number of ways, assuming that the cans aren't actually distinguishable.

So if 120 is overcounting, how many times have we overcounted?

If there were 120 ways of ordering 5 distinct items

Fixed overcounting: Maybe there are 5 orderings that break the rule, and we just need to find and subtract theses.
Multiple overcounting: E.g. for double counting, for every unique outcome, it's shown up exactly twice in our counting of 120.

It turns out in this case we've done the second one!

We have 120 outcomes of our *distinct* ones and zeroes like this:
![[Pasted image 20240705175525.png]]
For every distinct outcome, we could (eg) *swap* these distinct ones, which would have the same indistinct ordering.
- Whaat about for those zeros on the right? How many ways are there to order these 0s? Meaning how many *permutations* are there of 000? The answer for that is 3!.

120 / (2 * 3!) = 10

Let's get away from intuition:
- We want to think of a two-step process that create all the permutations of distinct objects

Step 1: Make all the permutations where some of our objects are indistinct
- All the permutations where 0s are indistinct
- All the permutations where 1s are indistinct

![[Pasted image 20240705175919.png]]
We actually want to find the middle term though, so
![[Pasted image 20240705175949.png]]
How many permutations there are, imagining they're all distinct, divided by the number of permutations of just the distinct objects

This is our
120 / (3! * 2!)


Let's test our knowledge! How many lettering orderings are possible for the string MISSISSIPPI?

Step 1: How many orderings are possible if we assumed that the objects were distinct?
- We choose one of the letters, then choose one of the remaining letters, then choose one of the remaining letters, etc.
|MISSISSIPPI| = 11 characters long
- 11!

Step 2: For every class (?) of object in the sequence, how many permutations are there?
- For the 1 M: 1
- For the 4 I's: 4!
- For the 4 S's: 4!
- For the 2 P's: 2

So...

`11! / (2*4!*4!)` = 34,650

----

![[Pasted image 20240705180817.png]]
- In this case, it just seems that one of six numbers has to be reused

Thinking: I think we could rephrase the problem a little bit. For any 5-length sequence from a pool of 5 numbers, the 6'th item we then select to *extend* that sequence has to be any of the 5 remaining numbers (there are 5 options).
So I think this could just be `5*5!` (Edit: Oops, I forgot something!)

A first good step for counting problems is to start writing out example outcomes, and explain out loud what you're doing as you generate outcomes.
- "Choose some permutation of the five numbers"
- "Choose the number that you want to repeat"
- "Choose the slot in the sequence that you want to stick the repeated number in."

And there are 5 choices for the double-digit
And the sequence that we generate is going to have 4 distinct and 2 indistinct numbers
6! would overcount the fact that 2 are indistinct!
![[Pasted image 20240705181448.png]]
So it's 6! divided by 2! to deal with the overcounting
And there are 5 different numbers we could choose, so: `5 * (6!/2!) = 1800`


With 5 books, think: How many ways are there to choose 3 of them?
- `5*4*3` = 60

![[Pasted image 20240705181648.png]]
![[Pasted image 20240705181909.png]]

![[Pasted image 20240705182015.png]]


Let's say there are 6 distinct books, and I want to choose 3? (So the ordering doesn't matter).
The answer here is just $6 \choose 3$ = 6!/3!3! = 20
- Get all my books in a line
- Draw a cutoff at 3
- Permute each group
Or... just recognize it as a combination problem, where you're choosing some unordered collection of distinct objects from a large pool!


![[Pasted image 20240705183638.png]]
Saw I have a deck of cards -- how many unique hands of 5 cards are there in a deck of 52 cards? (So the ordering of the hand doesn't matter).

Can we could how many different hands of cards are possible?
$52 \choose 5$


![[Pasted image 20240705184545.png]]
If you have distinct objects being put into distinct buckets, it's just a power.
$r * r * r * r$
r is the number of distinct buckets we have
n is the number of distinct items we have


But now let's talk about *indistinct* objects being put into buckets!
- If we had 4 items, and we thought the answer was $buckets^4$, we would be wrong from overcounting!

Let's put those 4 diet coke cans into 3 distinct buckets.
We can make this easier by thinking about (physical) dividers.

If we consider coke cans to be .
And dividers to be |

.|..|.
This describes a situation where one can is in bucket 1, 2 in bucket 2, and 1 in bucket 3
||....  
This describes a ... where all four cans are in bucket 3

We can think of any assignment of cans into buckets by thinking about ==some ordering of the coke cans and the dividers.==

So we have 4 coke cans and 2 dividers. 6 objects in total.

Are there 6! different orderings? 
- NO! Because we have 4 indistinct coke cans!

$6!/4!2!$

Using this from earlier
![[Pasted image 20240705185149.png]]
Isn't it interesting that the number 3 (the number of buckets) didn't shop up? We used 2!
If you have 3 buckets, we have 2 dividers.


![[Pasted image 20240705185310.png]]

![[Pasted image 20240705185319.png]]


































