---
aliases:
  - Recall-Oriented Understudy for Gisting Evaluation
---
In contrast to [[BLEU]], ROGUE is *recall-oriented* metric counting the number of words in the *reference* distribution that also appear in the *output*. It's typically used to assess automatic [[Summarization]] tasks.
Generally, there isn't good correlation with how humans evaluation fluency, particularly with tasks that require creativity and diversity -- outputs can have zero n-gram overlap with the reference but yet be a good response.

There are several ROGUE variants. ROGUE-N is most similar to BLEU, in that it also counts the number of matching n-grams between the output and reference.

![[Pasted image 20240525132824.png]]

Rogue Score
![[Pasted image 20240401235401.png]]
A simple metric for evaluating the output of a model by comparing it to a reference output. ==It works by simply counting the number of words== (or the number of n-grams, for ROGUE-N) in the ==reference output that *also* occur in the model's generated output==.

Benefits:
- A simple metric that's easy to compute, and correlates positively with human evaluation (When it works).

Drawbacks:
- ==Simply relying on lexical overlap might miss equally-good generations that are phrased differently!==
- ==It might also reward texts that have a large portion of common text, but actually have the inverse meaning!==

![[Pasted image 20240525134903.png]]

![[Pasted image 20240604163450.png|400]]
Above: By just taking the question and copying it many times in the answer, we get a ROGUE score that beats many *actual* systems (at question answering), and comes near to human answers! You can't just blindly trust ROGUE scores, they're... a pretty bad evaluation metrics, along with BLEU!


Variants: 
- ROGUE-N: Measures the number of matching n-grams between the model-generated text and a human-produced reference.
- ROGUE-1 Precision, Recall, and F1 scores: Precision is the ratio of the number of unigrams in C that also appear in R, over the number of unigrams in C. Recall is the ratio of the number of unigrams in R that also appear in C, over the number of unigrams in R. F1 score is computed from the other two metrics using the standard F1-score formula.
- ROGUE-2: Considering two-grams
- **ROGUE-L**: Based on the longest common subsequence (LCS) between our model output and reference; i.e. the longest sequence of words that are *not necessarily consecutive, but still in order* that is shared between both. 
- **ROGUE-S**: Allows us to add a degree of leniency to the n-gram matching performed with ROGUE-N and ROGUE-L; ROGUE-S is a skip-gram concurrence metric, letting us search for consecutive words from the reference text that appear in the model output but are separated by one or more other words.

Comparison with [[BLEU]]:
- BLEU focuses on precision: How much the words (or n-grams) in the candidate model outputs appear in the human reference.
- ROGUE focuses on recall: How much the words (or n-grams) in the human references appear in the candidate model outputs.

