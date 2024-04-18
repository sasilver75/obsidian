February 9, 2023 -- [[Meta AI Research]]
Link: [Toolformer: Language Models Can Teach Themselves to Use Toools](https://arxiv.org/abs/2302.04761)

One of the earlier LLM tool use papers by someone other than OpenAI (see: [[WebGPT]])

Abstract
> ==Language models (LMs) exhibit remarkable abilities== to solve new tasks from just a few examples or textual instructions, especially at scale. They also, ==paradoxically, struggle with basic functionality==, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that ==LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds==. We introduce Toolformer, a model ==trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction==. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a ==Q\&A system, two different search engines, a translation system, and a calenda==r. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.


Youtube Commenter: A big downside of this approach that you have to finetune the LLM each time you want to add a new tool, and finetuning is expensive/complicated -- it might better to use special languages like `sglang` or `guidance`.

![[Pasted image 20240415001134.png|400]]


