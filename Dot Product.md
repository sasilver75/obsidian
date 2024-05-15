Resources:
- [VIDEO: 3b1b Essence of Linear Algebra: Dot Products and Duality](https://www.youtube.com/watch?v=LyGKycYT2v0)

The sum of element-wise products of two same-dimensioned vectors.

$a \cdot b = a_1b_1 + a_2b_2 + a_3b_3 + ... a_nb_n$ 

The geometric interpretation is measuring how much one vector extends in the direction of the other; the dot product can be expressed in terms of the magnitudes of the vectors and the cosine of the angle $\theta$ between them:

$a \cdot b = ||a||\ ||b|| cos(\theta)$ 

Geometric Properties:
- Orthogonality: When $a \cdot b = 0$, we know that vectors are orthogonal to eachother; this is because $cos(90) = 0$.
- Angle measurement: The cosine of the angle between two vectors can be found using the dot product definition and a little algebra:
	- $cos(\theta) = (a \cdot b)/(||a||\ ||b||)$ 
- Projection: The dot product also relates to the projection of one vector onto another; The scalar projection of a onto b is given by ($a \cdot b$)/($||b||$) -- this represents the length of the projection of a in the direction of b.
- Magnitude Influence: If the vectors are in the same direction, then the dot product is maximal and positive. If the vectors are in opposite direction, then the dot product is maximal in magnitude but *negative*.
- 