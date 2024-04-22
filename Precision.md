---
aliases:
  - Positive Predictive Value
---
Precision can be seen as a measure of *quality*, and [[Recall]] as a measure of *quantity*.

Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned).

==Precision = TruePositives / (TruePositives + False Positives)==
Of all the times you say 'yes,' how many times were *actually* 'yes'?
  
![{\displaystyle {\text{Precision}}={\frac {\text{Relevant retrieved instances}}{\text{All retrieved instances}}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/f2fe5aa3d0e91f91abc0ead472c59737af6c47c0)

 Remember: [[Precision]] is the number of true positive results divided by all samples predicted to be positive, including those not predicted correctly. "Of all the times you say 'yes', what percentage are you correct?"

![[Pasted image 20240420220007.png]]