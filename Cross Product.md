Resources:
- [BLOG: 3b1b Cross Product](https://www.3blue1brown.com/lessons/cross-products)

The Cross Product is a way to multiply vectors 

If we have two vectors $\vec{v}$ and $\vec{w}$, we can think about the parallelogram that they span out:
![[Pasted image 20240603163807.png]]
The cross product (the ==area of this parallelogram==) is written as $\vec{v}\times\vec{w}$
- (There's a small extra consideration: We need to consider orientation of the vectors! If $\vec{w}$ is a counterclockwise rotation away from $\vec{v}$, then $\vec{v}\times\vec{w}$ is positive and equal to the area of the parallelogram, but if $\vec{w}$ is a clockwise rotation away from $\vec{v}$, the cross product is -1 *times* the area of that parallelogram)
	- ((I think the language used above in terms of which vector is rotated is a little ambiguous; the pictures are correct.))
![[Pasted image 20240603163817.png]]Above: A (negative) Cross Product

Notice that ==order matters==; If you swap $\vec{v}$ and $\vec{w}$, and instead consider $\vec{w}$ cross $\vec{v}$, the cross product becomes the negative of whatever it was before.

To actually calculate this, for the 2d cross product of $\vec{v}\times\vec{w}$ , we write the coordinates of $\vec{v}$ as the first column of a matrix, and the coordinates of $\vec{w}$ as the second column, and then compute the [[Determinant]]:
![[Pasted image 20240603164252.png]]
This is because a matrix whose columns represent $\vec{v}$ and $\vec{w}$ corresponds with a linear transformation that moves the [[Basis Vector]]s $\hat{i}$ and $\hat{j}$ to $\vec{v}$ and $\vec{w}$.
- Determinants are all about measuring how areas change due to transformations, and the prototypical area to look at is the unit square resting on  $\hat{i}$ and $\hat{j}$. 
- After the transformation, the square turns into the parallelogram we care about -- so the determinant, which generally measures the factor by which areas are changed, gives the area of this parallelogram, since it evolved from a square starting with area 1.