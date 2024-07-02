Resources:
- Blog: [Eugene Yan's Evaluation Blog Post](https://eugeneyan.com/writing/evals/)
- Blog: [Anthropic out out a call for new Evals that includes criteria on what makes a good evaluation](https://www.anthropic.com/news/a-new-initiative-for-developing-third-party-model-evaluations)

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

Overview
- Classification: Recall, Precision, ROC-AUC, Separation of Distributions
- Summarization: Factual consistency via NLI, Relevance via reward modeling
- Translation: Measure quality via chrF, BLEURT, COMET, COMETKiwi
- Toxicity: Test with adversarial prompts from RealToxicityPrompts and BOLD
- Copyright: Test with text from popular books and code


Classification:
- [[Recall]]
- [[Precision]]
- [[ROC-AUC]]
- [[PR-AUC]]
- Separation of distributions

Summarization
- To evaluate abstractive summaries, Kryscrinski et al (2019) proposed four key dimensions:
	- Fluency: Are sentences well-formed and easy to read + good grammar?
	- Coherence: Does the summary as a whole make sense? Well structured and logically organized.
	- Consistency: Does the summary accurate reflect the content of the source document.
		- Can be measured via an [[Natural Language Inference|NLI]] model, which return probabilities for entailment, neutral, and contradiction. 
	- Relevance: Does the summary focus on the most important aspects of the source document? It should include key points and exclude less relevant details.
- Relevance via reward model
	- *Learning to summarize from human feedback* (Sep 2020, Stiennon et al.), the predecessor of InstructGPT, trained a RM to evaluate abstractive summaries of Reddit posts.
- Length checks
	- Whether the model can follow instructions and n-shot examples to generate summaries that meet a word or character limit.

Translation
- Quality measures via [[chrF]]
	- Character n-gram F-score is similar to BLEU but operates at the character level instead of the word level. It's the second most popular metric for machine translation, with several advantages over BLEU.
- [[BLEURT]]
	- I believe this is a model that predicts a collection of related scores ([ [[BLEU]]]], [[ROUGE]], [[BERTScore]], etc.)
- [[COMET]]
- COMETKiwi

Copyright
- Exact regurgitation (compute the length of the longest common subsequence between the output and reference, normalized by the length of the input prompt)
- Near-exact reproduction (the edit distance and edit similarity between the output and reference, normalized by the length of the input prompt)

Toxicity
- Proportion of toxic generations on regular and toxic prompts



![[Pasted image 20240528105840.png]]
This is what Evals are really for (Bryan Bischof)

---

From Anthropic:
### Principles of good evaluations

[Developing great evaluations is hard](https://www.anthropic.com/news/evaluating-ai-systems). Even some of the most experienced developers fall into common traps, and even the best evaluations are not always indicative of risks they purport to measure. Below we list some of the characteristics of good evaluations that we’ve learned through trial-and-error:

1. **Sufficiently difficult:** Evaluations should be relevant for measuring the capabilities listed for levels ASL-3 or ASL-4 in our Responsible Scaling Policy, and/or human-expert level behavior.

2. **Not in the training data:** Too often, evaluations end up measuring model memorization because the data is in its training set. Where possible and useful, make sure the model hasn’t seen the evaluation. This helps indicate that the evaluation is capturing behavior that generalizes beyond the training data.

3. **Efficient, scalable, ready-to-use:** Evaluations should be optimized for efficient execution, leveraging automation where possible. They should be easily deployable using existing infrastructure with minimal setup.

4. **High volume where possible:** All else equal, evaluations with 1,000 or 10,000 tasks or questions are preferable to those with 100. However, high-quality, low-volume evaluations are also valuable.

5. **Domain expertise:** If the evaluation is about expert performance on a particular subject matter (e.g. science), make sure to use subject matter experts to develop or review the evaluation.

6. **Diversity of formats:** Consider using formats that go beyond multiple choice, such as task-based evaluations (for example, seeing if code passes a test or a flag is captured in a CTF), model-graded evaluations, or human trials.

7. **Expert baselines for comparison:** It is often useful to compare the model’s performance to the performance of human experts on that domain.

8. **Good documentation and reproducibility:** We recommend documenting exactly how the evaluation was developed and any limitations or pitfalls it is likely to have. Use standards like the Inspect or the METR standard where possible.

9. **Start small, iterate, and scale:** Start by writing just one to five questions or tasks, run a model on the evaluation, and read the model transcripts. Frequently, you’ll realize the evaluation doesn’t capture what you want to test, or it’s too easy.

10. **Realistic, safety-relevant threat modeling:** Safety evaluations should ideally have the property that if a model scored highly, experts would believe that a major incident could be caused. Most of the time, when models have performed highly, experts have realized that high performance on that version of the evaluation is not sufficient to worry them.