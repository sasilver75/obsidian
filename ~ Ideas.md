
# Blog Ideas
- Embeddings (Different techniques, e.g. Matryoshka, Applications)
- Tokenizers (BPE, SentencePiece, WordPiece, ...)
- Synthetic Data (Self-Instruct, Evol-Instruct, OSS-Instruct, Orca, Orca2, Genstruct, Rephrasing the Web etc.)
- Decoding Strategies (Greedy, TopK, TopP, MCTS-reward-model-guided, LaMBDA classifier decoding)

# Project Ideas
- 
# Research Ideas
- The idea of speculative decoding, where you have a smaller model running in parallel to the larger model -- is there a way of doing something like RAG in this manner? Where you're increasing the speed of rag by speculatively retrieving documents that might be needed in the next (eg) paragraph of text?
- Question@JaredKaplan re: CAI: "Why use a scalar for the reward as opposed to anything else?" -> "Interesting research questions; could imagine a bunch of functions applied to the reward; imagine punishing bad behavior more extremely than good behavior, or changing the way that you sample. We've mostly done the simplest thing, but there's interesting research to be done on variations."
- [[Stanford Human Preferences|SHP]] uses numbers-of-upvote heuristics on Reddit data to construct a synthetic preference dataset. Is there a way to use some other heuristic (eg some classifier ensemble) (maybe combined with some rewriting) to create a similar dataset that captures some of the false negatives using semantic understanding?
	- (Oh, is this already what AlpacaFarm, UltraFeedback, Kim et al. 2023, Xu et al., 2023) do?
- Mixture of depth says that we can use smaller networks (iirc) to predict "Easier" tokens. Matryoshka lets us have these variable-dimensional versions of vectors; is there some sort of way of doing something similar with model parameters (some MRL-similar loss of some sort) so that we can be infinitely (?) variable in the amount of compute that we put towards a token?


https://x.com/thesephist/status/1734966611814289756?s=20

- Automatic Commit Message Generation
- Use the streaming LLMs with attention sinks paper to summarize some sort of live data feed, like financial or news information.
- As per [this](https://youtu.be/TIqf4LMNCjU?si=lTLl_4ft-TmCpMNr) video, distillation is usually used to create a smaller model with a similar performance. Can you use distillation (by providing higher quality labels; either distributions or rationales) to train successively better versions of the same size model? What about slightly increasing capacity models? Do you need more capacity to benefit from the slightly better labels?
- Sort of like how yikes in the Latent Space paper club used OpenInterpreter to scrape a bunch of related papers into a Weaviate rag chatbot (Verba), and then used it as a sidecar to the presentation.... could you do the same thing for a powerpoint presentation, for instance? Extract topics, etc from the slides, find relevant documents ingest them, and people can answer questions they have about the presentation from a little sidecar Q&A bot while the presentation is still going, so they can have better understanding.

![[Pasted image 20240313115215.png]]

- A chrome extension that lets you comment on every page on the internet. The comment section of the internet. Bootstrap it with LLM outputs.