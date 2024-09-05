---
aliases:
  - Bilingual Evaluation Understudy
---
A precision-based metric (in contrast to [[ROUGE]]'s *recall-based metric*) that counts the number of n-grams in the generated output that also show up in the reference, and then divides it by the total number of words in the output. Remains a popular metric due to its cost-effectiveness. Typically used for [[Machine Translation]].

Generally, there isn't good correlation with how humans evaluation fluency, particularly with tasks that require creativity and diversity -- outputs can have zero n-gram overlap with the reference but yet be a good response.

First, precision for various values of $n$ is computed
![[Pasted image 20240525131636.png]]
$Count_{clip}(n-gram)$ is clipped by the maximum number of times an n-gram appears in any corresponding reference sentence.

$Count_{clip}(Ngram)$ = $min(matched\: Ngram\: count, max_{r\in{R}}(Ngram\: count\: in\: r))$ 

Once we compute this precision at various $n$, a final BLEU-N score is computed as the geometric mean of all of the $precision_n scores.

But since precision relies solely on n-grams and doesn't consider the length of the generated output, an output containing just one unigram of a common word (like a stop word) would achieve perfect precision. This can be misleading, encouraging outputs that contain fewer words to increase BLEU scores. To counter this, a *==brevity penalty==* is added to penalize excessively-short sentences.

![[Pasted image 20240525132258.png]]

Thus the ==final formula== is

$BLEU-N = BP \cdot exp(\sum_{n=1}^NW_nlog(precision_n))$ 

==Problems/Limitations==
- Simply relying on lexical overlap might miss equally-good generations that are phrased differently!
- It might also reward texts that have a large portion of common text, but actually have the *inverse* meaning!
- In BLEU, all words/n-grams are treated as equally relevant!
- Scores themselves are meaningless; the absolute value is not informative!
- Human translators also score low on BLEU


![[Pasted image 20240411201653.png]]


Note:
A simple precision-based metric (like the one below, which is *NOT* BLEU) has a problem, which is that it doesn't consider word order. BLEU is basically a modification of a precision-based metric that includes bi-grams, tri-grams, and four-grams, in addition to unigrams, to address ordering of words.
![[Pasted image 20240604162711.png]]

---

![[Pasted image 20240605203357.png]]
![[Pasted image 20240605203404.png]]
Above: I think p refers to the generated sentence, and r to the source sentence. $|p|$ is the *length* of the generated sentence.



