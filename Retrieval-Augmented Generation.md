---
aliases:
  - RAG
---
May 22, 2020
[[Meta AI Research]], UCL, NYU -- Authors include [[Douwe Kiela]]
Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
#zotero 
Takeaway: ...

Note that "RAG" is both a specific model/technique as described in this paper, but it's become a general term as well for retrieving text information and placing it into the prompt of a language model, though in those cases, it's often the case that the *retriever*  and *reader* aren't jointly trained.

----

The authors note that there are many shortcomings of the "End-to-End LLM Strategy", including hallucinations, inability to update/edit information, and inability to prove provenance of responses. By incorporating retrieval over a knowledge store, they hope to use the parametric model for "understanding," and the knowledge store (Wikipedia, in this case) for "knowledge," in an effort to ameliorate the three limitations above.

Notes:
- 

Abstract
> Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, ==their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures==. Additionally, ==providing provenance== for their decisions and ==updating their world knowledge== ==remain open research problems==. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. ==We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia,== accessed with a pre-trained neural retriever. ==We compare two RAG formulations==, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.









![[Pasted image 20240108230651.png]]