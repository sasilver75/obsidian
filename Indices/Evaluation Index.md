Note: Even with recent benchmarks like MMLU, the same model can get significantly different scores based on the evaluation implementation.
![[Pasted image 20240525135012.png|300]]
Above: HELM and EleutherAI implementations find that the same example can have different prompts across various providers.

Furthermore, the evaluation approach differed across all three benchmarks:
- Original MMLU: Compares predicted probabilities on the answers only (A, B, C, D)
- HELM: Uses the next token probabilities from the model and picks the token with the highest probability, even if it’s _not_ one of the options.
- EleutherAI: Computes probability of the full answer sequence (i.e., a letter followed by the answer text) for each answer. Then, pick answer with highest probability.

![[Pasted image 20240525135120.png]]

As a result, even for the same eval, both absolute scores and model ranking can fluctuate widely depending on eval implementation. This means that model metrics aren’t truly comparable—even for the same eval—unless the eval’s implementation is identical down to minute details like prompts and tokenization. Similarly, the author of QLoRA found MMLU overly sensitive and concluded: “[do not work with/report or trust MMLU scores](https://twitter.com/Tim_Dettmers/status/1673446047266504704)”.

----

From: [LLM Task-Specific Evals that Do and Don't Work](https://eugeneyan.com/writing/evals/)

If you've ran off-the-shelf evaluation for your tasks, you might have found that most don't work. They barely correlate with application-specific performance, and aren't discriminative enough to use in production.

Classification:
- [[Recall]]:
- [[Precision]]:
- [[Machine Translation]]:
- Copyright:
- Toxicity:

