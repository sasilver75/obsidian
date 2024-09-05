May 24, 2023 -- University of Washington and [[Allen Institute]] (AI12)
Paper: [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334)
See similar, prior work: [[Toolformer]], [[WebGPT]]

References:
- Video: [Teaching LLMs to use Tools at Scale (Stanford MLSys #98)](https://www.youtube.com/watch?v=WAvO8FTDJ8M&list=WL&index=121)
	- "The project has continued to evolve after the paper; at this point (May 2024), the project supports > 60,000 tools. It's a robust project being used at many enterprises (Apple, Cisco, NVDA, Microsoft, etc.). Many other people have taken the recipe for Gorilla and trained their own models."

Key ideas:
- ==Retriever-aware Training== (RAT) (How do we mix Fine-Tuning and Retrieval? The convention wisdom is that finetuning is for modifying behavior, and retrieval is for introducing new knowledge... but we found that finetuning is remarkably effective for both behavior and knowledge, and that retrievers are needed to data freshness, but are often inaccurate (~70% recall@k<5))
	- The big idea of RAT is to fine-tune the model to use or *ignore* retrieved context if it's not relevant. 
	- EG if the retriever says that we should use Resnet 79, the LLM is going to reason that it's never seen Resnet 79 before, so the retriever is probably being noisy, and I'll ignore this retrieval and use my parametric knowledge to return (eg) Resnet 50.
	- The key idea is just to introduce both *correct* as well as *incorrect* retrieval results during instruction-finetuning, ensuring that the model is robust to low-quality retrieval.
- Measuring hallucination
	- Normally, if you were to ask "how much does GPT-4 hallucinate," the response is something like "Oh, less than GPT-3.5." How can we quantify this? What are different degrees of hallucination?
	- We can define hallucination for API calls, though! We'll use Abstract Syntax Tree (AST) sub-tree matching.
		- When the model creates some code for an API call using the torch package, we create an AST out of the generated API call, and we also have available all of the API calls that are available in the library. Then we can ask the question: Is the the tree that my LLM generated a subset of the possible trees that could be generated? If so, we claim it's not hallucinated.
			- Note that just because something isn't hallucinated doesn't mean it's *accurate*. Hallucinations are when we run an API call and we get a 404 error, but if the API call runs and doesn't do what we want it to do, we say that it's not a hallucination, but isn't accurate. 
----


A 70B parameter LLM trained to use RAG and API tools. Trained on a dataset of over 5B tokens, and able to use 1575 (?) tools.

Abstract
> Large Language Models (LLMs) have seen an impressive wave of advances recently, with models now excelling in a variety of tasks, such as mathematical reasoning and program synthesis. However, ==their potential to effectively use tools via API calls remains unfulfilled==. This is a challenging task even for today's state-of-the-art LLMs such as GPT-4, largely due to their inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call. We release ==Gorilla, a finetuned LLaMA-based model that surpasses the performance of GPT-4 on writing API calls==. When ==combined with a document retriever==, Gorilla demonstrates a strong capability to adapt to test-time document changes, enabling flexible user updates or version changes. It also ==substantially mitigates the issue of hallucination, commonly encountered when prompting LLMs directly.== To evaluate the model's ability, we introduce APIBench, a comprehensive dataset consisting of HuggingFace, TorchHub, and TensorHub APIs. The successful integration of the retrieval system with Gorilla demonstrates the potential for LLMs to use tools more accurately, keep up with frequently updated documentation, and consequently increase the reliability and applicability of their outputs. Gorilla's code, model, data, and demo are available at [this https URL](https://gorilla.cs.berkeley.edu/)


![[Pasted image 20240415001406.png]]
