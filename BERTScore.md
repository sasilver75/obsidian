An embedding-based metric that uses cosine similarity to compare each token or n-gram in the generated output with the reference sentence.

There are three components to BERTScore:
- Recall: Average cosine similarity between each token in the *reference* and its closest match in the generated output.
- Precision: Average cosine similarity between each token in the *generated output* and its nearest match in the reference.
- F1: Harmonic mean of recall and precision

![[Pasted image 20240525133334.png]]

BERTScore is useful because it can account for synonyms and paraphrasing; similar metrics like [[BLEU]] and [[ROUGE]] can't do this, due to their reliance on exact matches. BERTScore has been shown to have better correlation (with human evaluations) for tasks like image captioning and machine translation.