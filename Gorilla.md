May 24, 2023 -- University of Washington and [[Allen Institute]] (AI12)
Paper: [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)
See similar, prior work: [[Toolformer]], [[WebGPT]]

A 70B parameter LLM trained to use RAG and API tools. Trained on a dataset of over 5B tokens, and able to use 1575 (?) tools.

Abstract
> Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, ==their potential to effectively use tools via API calls remains unfulfilled==. This is a challenging task even for today's state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call. We release ==Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls==. When ==combined with a document retriever==, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also ==substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly.== To evaluate the model's ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs. Gorilla's code, model, data, and demo are available at [this https URL](https://gorilla.cs.berkeley.edu/)


![[Pasted image 20240415001406.png]]
