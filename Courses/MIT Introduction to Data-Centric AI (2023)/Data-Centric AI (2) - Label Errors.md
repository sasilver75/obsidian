https://www.youtube.com/watch?v=AzU-G1Vww3c&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=2

---

This lecture is gonna be pretty fast-paced and jam-packed compared to last time.

![[Pasted image 20240626003833.png]]
Here are some label errors from the [Label Errors](https://labelerrors.com) site.
- Label issues can take many forms!
	- ==Correctable== (Where there's only one clear label in the dataset as far as we can tell, and the given label is just wrong. This is the simplest and clearest case, and we'll focus in this lecture on how to detect these.)
	- ==Multi-Label== (If you have a dataset that you intend to be single layer, but two of your classes are present in an image)
	- ==Neither== (Potentially out-of-distribution; that's an L, but this is a digits dataset! Maybe something is clearly two people, but the two possible labels are rose and apple).
	- ==Non-agreement== (Just "hard" examples)

Agenda
1. Label issues (kinds, why they matter, etc.)
2. Noise processes and types of label noise (How do they become erroneous?)
3. How to find label issues
4. Mathematical intuition for why the methods work
5. How to rank data by likelihood of having a label issue





