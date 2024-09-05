April 16, 2022 (1 year after [[Natural Instructions]] -- Huge number of contributors from an institutions, led (?) by [[Allen Institute]]
[Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks](https://arxiv.org/abs/2204.07705)

- A ==crowd-sourced== (from humans) collection of instruction-data based on existing NLP tasks and simple synthetic tasks..
- Includes ==5M== examples across in 1616 tasks across ==76 distinct task types== in 55 languages.
- Compared to [[Natural Instructions]], the *instructions are simplified* -- the consist of a task definition and positive/negative examples, with explanations.

Abstract
> How well can NLP models generalize to a variety of unseen tasks when provided with task instructions? To address this question, we first introduce ==Super-NaturalInstructions==, a ==benchmark of 1,616 diverse NLP tasks and their expert-written instructions==. Our collection covers ==76 distinct task types==, including but not limited to classification, extraction, infilling, sequence tagging, text rewriting, and text composition. This large and diverse collection of tasks ==enables rigorous benchmarking of cross-task generalization under instructions== -- training models to follow instructions on a subset of tasks and evaluating them on the remaining unseen ones. Furthermore, we build Tk-Instruct, a transformer model trained to follow a variety of in-context instructions (plain language task definitions or k-shot examples). Our experiments show that Tk-Instruct outperforms existing instruction-following models such as InstructGPT by over 9% on our benchmark despite being an order of magnitude smaller. We further analyze generalization as a function of various scaling parameters, such as the number of observed tasks, the number of instances per task, and model sizes. We hope our dataset and model facilitate future progress towards more general-purpose NLP models.

![[Pasted image 20240424003436.png]]

