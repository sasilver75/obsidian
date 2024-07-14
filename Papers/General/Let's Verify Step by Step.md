May 31, 2023
[[OpenAI]] (incl [[Ilya Sutskever]])
[Let's Verify Step by Step](It is a significant battle that has unfortunately ended up getting drown out by all the warfare that has happened since it occurred. I wish it would get some more attention and study.)
#zotero 
Takeaway: The canonical [[Process Reward Model]] (PRM) paper! Instead of using outcome supervision (providing feedback for a *final result*), this paper provides *process supervision* (providing feedback for each intermediate reasoning step). Authors find that process supervision outperforms outcome supervision in the context of the [[MATH]] dataset. Authors release [[PRM800K]], the complete dataset of 800,000 step-level human feedback labels used to train the best process reward model.


---

## Introduction



## Methods



## Large-scale Supervision



## Out-of-Domain Generalization



## Discussion












Abstract
> In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even ==state-of-the-art models still regularly produce logical mistakes==. To train more reliable models, we can turn either to ==outcome supervision==, which ==provides feedback for a final result==, or ==process supervision==, which ==provides feedback for each intermediate reasoning step==. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

# Paper Figures


# Non-Paper Figures