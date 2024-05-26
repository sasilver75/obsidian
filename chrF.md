---
aliases:
  - Character N-Gram F-Score
---
Similar to [[BLEU]] but operates at the character level instead of the word level. The second most popular metric for [[Machine Translation|MT]]; has several advantages over BLEU. It doesn't capture aspects of higher-level translation quality like fluency, coherency, and adequacy, but it's a solid eval to start with.

The idea is to compute the precision and recall of character n-grams between the machine translation and the reference translation.
- Precision (chrP) measures the proportion of character n-grams in the machine translation that match the reference.
- Recall (chrR) measures the proportion of character n-grams in the reference that are captured by the MT.
This is typically done for various values of $n$ (typically up to 6).

To combine chrP and chrR, we use a harmonic mean with $\beta$ as a parameter that controls the relative importance of precision and recall.
![[Pasted image 20240525222320.png]]
With $\beta=1$, precision and recall have equal weight; higher values of $\beta$ assign more importance to recall.

Benefit
- Doesn't require pre-tokenization, since it operates directly on the character level. Thsi makes it nice for languages with complex morphology or non-standard written forms.
- Computationally efficient, as it mostly involves string-matching operations that can be parallelized and run on CPU.
- Language-independent, and can be used to evaluate translations over many language pairs (this is an advantage over learned metrics like BLEURT and COMET, which need to be trained for each language pair).