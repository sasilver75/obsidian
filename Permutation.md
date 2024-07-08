References:
- Course Reader: [CS 109 Lecture 2 Combinatorics Notes](https://web.stanford.edu/class/archive/cs/cs109/cs109.1192/lectureNotes/2%20-%20Combinatorics.pdf)

A permutation is an *ordered* arrangement of objects.

==The number of unique orderings (permutations) of n *distinct* objects is $n!$==
- iPhones have 4 digit passcodes; suppose there are 4 smudges over 4 digits on the screen. How many distinct passcodes are possible? $4! = 24$.
- What if there were 3 smudges over 3 digits on a screen? We can solve this by making three cases, one for each digit that could be repeated. We might think this is $3 * 4!$, but we need to eliminate the double-counting of the permutations of identical digits (eg $ABC_1C_2$ and $ABC_2C_1$ should be considered the same passcode). We do this by dividing by the permutations of the indistinct objects. $3 * (4!/(2!*1!*1!)) = 36$ 
- If there were 2 smudges over two digits? This would mean either 1 digit is used 3 times, or 2 digits are used twice each:
	- $4!/(2!*2!) + 2*(4!/3!*1!) = 6+8 = 14$ 
	- There are two cases where 1 of the two numbers are repeated (3As and a B, or 3Bs and an A), which is why we have that 2x. In contrast, there's only one way that each of the two numbers is used twice (2As and 2Bs).

==The number of unique orders (permutations) of n distinct objects is== $n!/(n_1!n_2!...n_r!)$ 

![[Pasted image 20240705175919.png]]
![[Pasted image 20240705175949.png]]

![[Pasted image 20240708141754.png]]
![[Pasted image 20240708141805.png]]
