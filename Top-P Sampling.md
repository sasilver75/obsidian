---
aliases:
  - Nucleus Sampling
---
Some good discussion at ~45:00 in this CS685 [lecture](https://www.youtube.com/live/WoJrlvu7ODI?si=7oeVAKibbhkmGDzp)

![[Pasted image 20240411131239.png]]

Compare with [[Top-K Sampling]]: Top-k sampling samples tokens from those k with the highest probabilities (renormalizing probabilities and sampling) until the specified number of tokens is reached. In contrast, Top-p sampling samples tokens from the set of tokens having cumulative probability over some specified threshold value.k

