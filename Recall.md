---
aliases:
  - Coverage
  - True Positive Rate
---
While [[Precision]] can be seen as a measure of *quality*, Recall as a measure of *quantity*.

Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned).

![{\displaystyle {\text{Recall}}={\frac {\text{Relevant retrieved instances}}{\text{All relevant instances}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3c5350d4d74f4e18962798289b91795e76140b4)

If our model has a recall of .11, it correctly identifies (has a positive response for) 11% of all of the records that *actually* represented tumors.

Remember: [[Recall]] is the number of true positive results divided by the number of all samples that *should* have been identified as positive. "Of all the times that something was *actually ground-truth true*, what percentage of the time did you correctly identify this?"



![[Pasted image 20240420220349.png]]