Resources:
- [BLOG: 3b1b Inverse Matrices, Rank, Null Space](https://www.3blue1brown.com/lessons/inverse-matrices)

Linear Algebra is useful for describing the manipulation of space, but it's broadly applicable because it lets us solve certain ==systems of equations==, where we have a list of variables that we don't know, and a list of equations relating them.

![[Pasted image 20240603174006.png|300]]

If you're lucky, your equations relating these variables might take a certain special form, where the only thing happening to each variable is that it will be scaled by some constant, and the only thing happening to each of these variables is that they're being added (or subtracted) to eachother.

We typically organize these systems of equations by throwing all the variables on the left (vertically lining up the common variables), and constants on the right:
![[Pasted image 20240603174113.png|300]]

This looks a little line matrix-vector multiplication 🤔
🙂💡 In fact, we can package up all the equations together into ONE vector equation:
![[Pasted image 20240603174215.png|300]]

![[Pasted image 20240603174254.png]]
Above: Think: This is what's actually being represented by our $A\vec{x} = \vec{v}$ equation! We have some vector $\vec{x}$ that's being transformed by linear transformation $A$, and the resulting vector is $\vec{v}$.

We can find our initial vector $\vec{x}$ by playing the transformation of $A$ *==in reverse==*, from the resulting $\vec{v}$, whose values we know.

This inverse transformation of A, "A inverse," is often denoted as $A^{-1}$.
- If A were a counterclockwise rotation by 90, then the inverse of A would be a clockwise rotation by 90.
- If A were a rightward shear pushing $\hat{j}$ one unit to the right, the inverse of A would be a leftward shear that pushed $\hat{j}$ one unit to the left.

Generally:

$A^{-1}(A\vec{v}) = \hat{v}$ 

and

![[Pasted image 20240603174817.png|200]]

There are computational methods to compute these inverse matrices. In the case of two-dimensions, there's a commonly taught formula that's very hard to remember -- just use a computer.





