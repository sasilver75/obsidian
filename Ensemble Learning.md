---
aliases:
  - Ensemble
  - Ensembling
---


A technique aiming to enhance accuracy and robustness in forecasting by merging predictions from multiple models, aiming to mitigate errors or biases that may exist in individual models by leveraging the collective intelligence of the ensemble.


Linear Interpolation
![[Pasted image 20240617150000.png]]
![[Pasted image 20240617150730.png]]

So which one should we pick, Linear or Log-Linear interpolation?
- ==Linear: "Logical OR"==
	- The interpolated model likes any choice that a model gives a higher probability.
	- Use models with models that capture different traits.
	- Necessary when any model can assign zero probability (eg models with different vocabularies; this is hard to do, think about whether you want to!)
- ==Log-Linear: "Logical AND"==
	- Only likes choices where all models agree
	- Useful when you want to have a model in the mix that helps restrict possible answers (eg a model averse to toxic language)
	- Your interpolation coefficients dont' need to be positive, they can be negative too! If you want to use a model as *negative evidence* that you want to remove