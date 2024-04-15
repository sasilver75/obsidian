
# Blog Ideas
- Embeddings (Different techniques, e.g. Matryoshka, Applications)
- Tokenizers (BPE, SentencePiece, ...)
- Synthetic Data (Self-Instruct, Evol-Instruct, Orca, etc.)

# Project Ideas


# Research Ideas
The idea of speculative decoding, where you have a smaller model running in parallel to the larger model -- is there a way of doing something like RAG in this manner? Where you're increasing the speed of rag by speculatively retrieving documents that might be needed in the next (eg) paragraph of text?


https://x.com/thesephist/status/1734966611814289756?s=20

- Automatic Commit Message Generation
- Use the streaming LLMs with attention sinks paper to summarize some sort of live data feed, like financial or news information.
- As per [this](https://youtu.be/TIqf4LMNCjU?si=lTLl_4ft-TmCpMNr) video, distillation is usually used to create a smaller model with a similar performance. Can you use distillation (by providing higher quality labels; either distributions or rationales) to train successively better versions of the same size model? What about slightly increasing capacity models? Do you need more capacity to benefit from the slightly better labels?
- Sort of like how yikes in the Latent Space paper club used OpenInterpreter to scrape a bunch of related papers into a Weaviate rag chatbot (Verba), and then used it as a sidecar to the presentation.... could you do the same thing for a powerpoint presentation, for instance? Extract topics, etc from the slides, find relevant documents ingest them, and people can answer questions they have about the presentation from a little sidecar Q&A bot while the presentation is still going, so they can have better understanding.

![[Pasted image 20240313115215.png]]

- A chrome extension that lets you comment on every page on the internet. The comment section of the internet. Bootstrap it with LLM outputs.