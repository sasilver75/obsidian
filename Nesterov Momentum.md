References:
- Video: [DeepBeen's Optimization for Deep Learning](https://youtu.be/NE88eqLngkg?si=rgi-jKviVsmPZyM-)

![[Pasted image 20240705221317.png]]

![[Pasted image 20240705221328.png]]
Above: $\eta$ is the learning rate, and $g(...)$ is the gradient $\nabla$ 
- See that the gradient is calculated *after* the addition of the velocity jump.

![[Pasted image 20240705221407.png]]
In other words, w jumps ahead to some "look-ahead point" and then takes the gradient step from there. The issue this attempts to solve is the one in which the gradient calculated before the velocity jump might not be the appropriate step to take *after* the velocity jump. By calculating gradient at the lookahead point, we mitigate this issue.


