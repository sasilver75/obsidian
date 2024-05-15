---
aliases:
  - Contrastive Iterative Negative Generation
---

November 10, 2022
[[Meta AI Research]], Princeton
Paper: [The CRINGE Loss: Learning what language NOT to model](https://arxiv.org/abs/2211.05826)
...
Takeaways: ...

---

Notes:
- ...

Abstract
> ==Standard language model training== employs gold human documents or human-human interaction data, and ==treats all training data as positive examples==. Growing evidence shows that even with very large amounts of positive training data, ==issues remain that can be alleviated with relatively small amounts of negative data== -- examples of what the model should not do. In this work, we propose a novel procedure to train with such data called the ==CRINGE loss (ContRastive Iterative Negative GEneration).== We show the effectiveness of this approach across three different experiments on the tasks of safe generation, contradiction avoidance, and open-domain dialogue. Our models outperform multiple strong baselines and are conceptually simple, easy to train and implement.