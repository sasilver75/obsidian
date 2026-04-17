A [[Space-filling Curve]] that's constructed recursively.

At each level, you subdivide a square into 4 quadrants and connect them in a U shape, then recurse into each quadrant, rotating/reflecting to keep the path continuous.

At infinite recursion, it fills every point in the square.

For GIS grid systems like [[S2 Geometry|S2]] or  [[H3]] or [[A5]], they're useful.
Their crucial useful property is that of ==locality preservation==: points that are close together in 2D space tend to have close positions on the 1D curve.
- This is not perfect, but it's much better than naive row-by-row scanning.

![[Pasted image 20260416181959.png|500]]


![[Pasted image 20260416183251.png]]

![[Pasted image 20260416182630.png|500]]



