



> BEIR is practically useless nowadays; it was useful as a zero-shot test set. Among models trained openly on MS MARCO _without_ BEIR contamination or feedback, ColBERTv2 is ahead of everything else. Nowadays, all the top models are build FOR BEIR/MTEB evals, and use (at best) the training splits of BEIR tasks (and more often than not, direct/indirect validation on BEIR test sets). I have a big text file with 50 papers that shows the ColBERT paradigm being 50-100x more data efficient and up to 15-30 points better in quality than single vector. Note: "Everything else" here refers to dense retrievers of the same size and budget/tricks; you CAN build expensive cross-encoders that do better than vanilla BERT-base ColBERTv2. SPLADE is a great competitor at mid-scale, but dense bi-encoders are generally not.
> - Omar Khattab, Jan 28 2024 (ColBERT): https://x.com/lateinteraction/status/1751661624539357550