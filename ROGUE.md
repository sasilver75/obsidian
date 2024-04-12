---
aliases:
  - Recall-Oriented Understudy for Gisting Evaluation
---
Rogue Score
![[Pasted image 20240401235401.png]]
A simple metric for evaluating the output of a model by comparing it to a reference output. ==It works by simply counting the number of words== (or the number of n-grams, for ROGUE-N) in the ==reference output that *also* occur in the model's generated output==.


==Simply relying on lexical overlap might miss equally-good generations that are phrased diffrently!==

==It might also reward texts that have a large portion of common text, but actually have the inverse meaning!==