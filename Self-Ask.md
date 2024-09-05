October 7, 2022 (1 month before ChatGPT, 7 months after CoT)
UW, [[Allen Institute|AI2]], [[MosaicML]], [[Meta AI Research]] -- Authors include [[Ludwig Schmidt]]
Paper: [Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)
Takeaway: ... 

---

Notes: 
- I like the definition of the ==compositionality gap==, which is how often models can correctly answer *all* sub-problems, but still *not* generate the overall solution.
- The authors introduce a new technique called [[Self-Ask]], in which the LM explicitly asks (then answers) itself followup questions before answering the initial question.

Abstract
> We investigate the ability of language models to perform ==compositional reasoning tasks where the overall solution depends on correctly composing the answers to sub-problems==. We measure how often models can correctly answer all sub-problems but not generate the overall solution, a ratio we call the ==compositionality gap==. ==We evaluate this ratio by asking multi-hop questions with answers that require composing multiple facts unlikely to have been observed together during pretraining==. In the GPT-3 family of models, as model size increases we show that the single-hop question answering performance improves faster than the multi-hop performance does, therefore the compositionality gap does not decrease. This surprising result suggests that while more powerful models memorize and recall more factual knowledge, they show no corresponding improvement in their ability to perform this kind of compositional reasoning.
> We then demonstrate how elicitive prompting (such as ==chain of thought==) ==narrows the compositionality gap by reasoning explicitly==. We present a new method, ==self-ask==, that further ==improves on chain of thought==. In our method, the ==model explicitly asks itself (and answers) follow-up questions before answering the initial question==. We finally show that self-ask's structured prompting lets us easily plug in a search engine to answer the follow-up questions, which additionally improves accuracy.