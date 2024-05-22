(Model measure)

==[[Perplexity]] is the reciprocal of the geometric mean of this sequence probability!==
	- ==Low perplexity== values indicate that a language model assigns *high* probability to textual sequences used for evaluation, and therefore fits the evaluation data *well*.
	- ==High perplexity== values indicate that a language model assigns *low* probability to textual sequences used for evaluation, and therefore fits the evaluation set *poorly*.

If we had an absolutely perfect LM that gave us a P(1) for the correct next word in the sequence, this perplexity would be 1 (because the model thinks there's only 1 path that's likely). So 1 is the absolute minimum perplexity we could get. The maximum perplexity we can get is the vocabulary size of our model (which is why it's interesting to know the vocabulary size when we ask about perplexity), in a case where we think every word in our vocabulary is equally likely (we're *very* perplexed, here).


![[Pasted image 20240521204100.png]]

![[Pasted image 20240521204130.png]]
The equation for perplexity is basically just exponentiating the negative log likelihood (per-token likelihood) on some held-out text corpus.

![[Pasted image 20240521204202.png]]
If we think about Perplexity; assume we were given a prefix "I want the" and we want to know how many different paths/next words the model thinks are equally likely, given the prefix.
- If the model had NO IDEA what cames after, then maybe it assigns roughly equal probability to aLL words in the vocabulary, meaning the model is very confused -- it's highly *perplexed*
- If the model is very good, it know that *dog*, *streets*, etc. are likely -- some small number of words. ==So you can view this as a sort of branching factor -- given a prefix, how many words does the model think are reasonable given the prefix?==

Above: What's the perplexity of a sentence of 10 random digits?
- 10!
- The probability of any sequence is (1/10)^N
- Perplexity then exponentiates this by -1/N
- This result in 10; it means that 10 numbers are equally likely given any prefix, given that this is a totally random model. 

In practice, we use negative log likelihood as our training signal as the thing we're trying to maximize

![[Pasted image 20240521204521.png]]

Se if we have the log probability of word i given previous context, we sum these log probabilities over the entire sequence (because when we use logs, we sum, rather than multiply), and then normalize by the number of words in the sequence; we're getting the average token-level probability of the next word under the language model.



![[Pasted image 20240521205317.png]]
What if in our test set we have some phrase that just never occurred in the training set? The probability of any sentence like these will be 0 -- even though that's probably not true!

![[Pasted image 20240521205636.png]]
We estimate probabilities as we usually do, but we *steal* some probability from every observed sequence in our dataset, and distribute that somehow across all of the zero-count entries in the table. 
There was a lot of research on smoothing strategies 10-20 years ago; We've pretty much stopped using N-Gram models, but variants of this kind of approach were used in SoTA language models even 10 years ago.

