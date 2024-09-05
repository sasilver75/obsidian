Resources: 
- [BLOG: 3b1b Determinants](https://www.3blue1brown.com/lessons/determinant)

The scaling factor by which a linear transformation changes areas.
- Determinants can be anything in the \[-$\infty, \infty$\] range, including 0.


![[Pasted image 20240603165222.png|300]]

One thing that's useful to understand about a given transformation is exactly how much a it stretches and squishes the area of a given region in space.

For example, the following matrix
$$ \begin{array}{cc} 3 &0 \\ 0 &2 \end{array} $$
scales $\hat{i}$ by a factor of 3, and scales $\hat{j}$ by a factor of 2. If we focus our attention on the 1x1 square whose bottom sits on $\hat{i}$ and whose left side sits on $\hat{j}$ ... then we see that after the transformation, this turns into a 2x3 rectangle.
- It turns out that if we know how much the area of that one unit square changes, we know how the area of *any* region in space changes (assuming a linear transformation). This is because in a linear transformation, grid lines remain parallel and even spaced.

![[Pasted image 20240603165242.png|300]]
Above: Determinants ==can make areas smaller too!==
- ==Determinants of 2D transformations can be 0== if they squash all of space onto a line, or even a single point, since the area of every region becomes 0 in both cases.

But wait, ==Determinants can be negative too==, right? So what does it mean to scale an area by a negative amount? It has to do with the idea of orientation.

Below, notice how the unit vector $\hat{j}$ is usually to the left of $\hat{i}$:
![[Pasted image 20240603165614.png|300]]

After a transformation like 
$$ \begin{array}{cc} 1 &2 \\ 3 &4 \end{array} $$
We can see that our unit vector $\hat{j}$ is to the RIGHT of $\hat{i}$:
![[Pasted image 20240603165715.png|300]]

If you think about 2D space as a sheet of paper, transformations like that one seem to *turn over the sheet* to the other side. Transformations that do this area said to invert the orientation of space! When this happens, determinants are negative (but the absolute value of the determinant still tells you the factor by which areas are scaled).

The formula for a 2x2 matrix:
![[Pasted image 20240603165859.png|300]]
Above intuition:
- Let's say that $b$ and $c$ were zero -- then the term $a$ tells you how much $\hat{i}$ is scaled in the x-direction, and the term $d$ tells you how much $\hat{j}$ is scaled in the y-direction.
- Loosely speaking, if both $b$ and $c$ are nonzero, the term $b \cdot c$ tells you how much this rectangle is stretched or squished in the diagonal direction.

Formula for a 3x3 matrix:
![[Pasted image 20240603170109.png|300]]
(It's not frankly important that I remember the formula for determinants of n-dimensional matrices; the essence is that they describe the degree to which space is transformed by a linear transformation.)

Note: ==If you multiply two matrices together, the determinant of the resulting matrix is the same as the product of the determinants of the two original matrices.== (Intuitively, this makes sense)
![[Pasted image 20240603170248.png|300]]
