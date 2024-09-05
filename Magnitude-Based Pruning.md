---
aliases: []
---
A method for selecting which neurons to prune.

![[Pasted image 20240628132904.png]]
Intuition: Prune the one with the lowest magnitude!

![[Pasted image 20240628133100.png]]

We can also apply it to prune whole rows of weights, in a form of coarse-grained structured pruning:
![[Pasted image 20240628133154.png]]
Here we're just doing the absolute value of the sum of each row.

![[Pasted image 20240628133222.png]]
Here's another version where we use the L2 norm, rather the the L2 norm used in the previous example.