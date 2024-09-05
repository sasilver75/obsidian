---
aliases:
  - LM Judge
---


See use in [[Constitutional AI]]

LLM judgements tend to be less noisy than human evaluation (due to lack of differing biases among annotators; the bias here is more systematic), but more biased. These biases include (among other political biases perhaps inherent in models):
1. [[Positional Bias]]: LLMs tend to favor responses in the first position. To mitigate this, we can evaluate the same pair of responses while swapping their order -- if the same response is preferred in both orders, we mark it as a win; else, it's a tie.
2. [[Verbosity Bias]]: LLMs tend to favor longer, wordier responses over more concise ones, even if the latter is clearer and of higher quality. A possible solution is to ensure that comparison responses are similar in length.
3. [[Self-Enhancement Bias]]: LLMs have a slight bias toward their own answers; To counter this, don't use the same LLM for evaluation task. An interesting question if you're using some smaller model is whether it was trained using (data) distillation on any (eg) GPT-4 outputs, and what effect that has on the judge's evaluations.

Compare: Pairwise Ranking, Direct Assessment
![[Pasted image 20240516223002.png|450]]
Above: I believe that "Reference-Free" evaluation is another term for Direct Assessment, where you don't need a "golden reference", and you can just assess the quality of the output based solely on the input prompt and the model's response.
- Note that reference-free evaluations can sometimes be used as [[Guardrails]] too, or vice versa.

