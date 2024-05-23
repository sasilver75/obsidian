Refers to instances where the language model generates information that's not grounded in data or reality. When the output is fabricated, incorrect, or unsupported by provided data. We call them hallucinations because the model so confidently (as always) asserts them.

Hallucinations can be mitigated by using higher-quality datasets (diverse, accurate, minimally biased, domain-specific), aligning into the model (eg by RLHF) the ability to *refuse* to answer questions that it doesn't know, or by giving the model access to external knowledge sources (RAG).


OpenAI in a blog post talked about: (Ref: Ep 11 Zeta Alpha)
- Open-domain hallucination
- Closed-domain hallucination

