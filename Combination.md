References:
- Course Reader: [CS 109 Lecture 2 Combinatorics Notes](https://web.stanford.edu/class/archive/cs/cs109/cs109.1192/lectureNotes/2%20-%20Combinatorics.pdf)

A combination is an *==unordered==* selection of $k$ objects from a set of $n$ *==distinct==* objects.
- Example: How many ways could I select $k$ people from a group of $n$ to receive cake?

$n! / (k!(n-k)!)$  , which can also be written as $n \choose k$

Fun fact: $n \choose k$ = $n \choose n-k$

This equation shows up so often that it gets its own notation, which is called the [[Binomial Coefficient]]

![[Pasted image 20240705182230.png]]
(Where these humans are distinct)

Let's say there are 6 distinct books, and I want to choose 3? (So the ordering doesn't matter).
The answer here is just $6 \choose 3$ = 6!/3!3! = 20
- Get all my books in a line
- Draw a cutoff at 3
- Permute each group
Or... just recognize it as a combination problem, where you're choosing some unordered collection of distinct objects from a large pool!

![[Pasted image 20240708141846.png]]




----
This isn't really about combinations, but I don't have another place to put this bucketing information about number of ways to put indistinguishable balls or distinguishable balls into buckets. There are more ways to arrange distinguishable items, because their individual identities matter, and swapping identical items leads to a new outcome.
![[Pasted image 20240708142036.png]]