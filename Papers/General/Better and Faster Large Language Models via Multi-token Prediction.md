April 30, 2024
[[Meta AI Research]], CERMICS Ecole des Ponts ParisTech, LISN Université Paris-Sacley
[Better and Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
#zotero 
Takeaway: ...

This paper came out a few months after the [[Medusa]] paper, and is similar in the sense that they both involve multi-token prediction.

----

## Introduction


## Methods


## Experiments on Real Data


## Ablations on Synthetic Data


## Why does it work? Speculation


## Related Work


## Conclusion


Abstract
> Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, ==at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk==. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3 times faster at inference, even with large batch sizes.


# Paper Figures

![[Pasted image 20240721234532.png|300]]

![[Pasted image 20240721234558.png|300]]

![[Pasted image 20240721234617.png|300]]

![[Pasted image 20240721234647.png|500]]

![[Pasted image 20240721234707.png|200]]


![[Pasted image 20240721234724.png|200]]

![[Pasted image 20240721234739.png|250]]

![[Pasted image 20240721234753.png|250]]

![[Pasted image 20240721234806.png|250]]

![[Pasted image 20240721234822.png|250]]





























