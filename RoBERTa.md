---
aliases:
  - Robustly Optimized BERT Approach
---
July 26, 2019
[[Meta AI Research]]
Paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

Addresses some of the known limitations with BERT, chiefly the observation that the BERT team did an admirably detailed, but still partial set of ablation and optimization studies. The RoBERTa team takes over, trying to do a more thorough exploration of the design space.
- Turns out we just scale the models bigger and nail the hyperparameters and we get a great BERT model!

At a meta level, this paper points to a shift in methodologies: RoBERTa team does a thorough examination of hyperparameters, but it's nowhere near the *exhaustive* hyperparameters sweeps before the BERT era, because it's just too expensive!

Abstract:
> Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, ==hyperparameter choices have significant impact on the final results==. ==We present a replication study of BERT pretraining== (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. ==We find that BERT was significantly undertrained==, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results ==highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements==. We release our models and code.

![[Pasted image 20240425145251.png]]
Above:
- BERT used a static masking approach, meaning they copied their training data some number of times, applying different masks to each copy. That set of copies of the dataset was then used repeatedly during epochs of training, so the same masking was seen multiple times by the model. We can get more diversity if we *==dynamically* mask examples== as we load individual batches, so that subsequent batches containing the same examples have different masking applied to them.
- For BERT, the inputs to the model were two concatenated document segments, which was crucial for their next-sentence prediciton task. RoBERTa incldued sentences that even span document boundaries. 
- Correspondingly, RoBERTa just ==dropped the NSP objective== on the grounds that it wasn't earning its keep.
- RoBERTa ==increased batch size== from 256 -> 2,000
- BERT used a WordPiece tokenizer, whereas RoBERTa used a character-level [[Byte-Pair Encoding|BPE]] algorithm.
- BERT trained only on BooksCorpus and English Wikipedia; RoBERTA added CC-News, OpenWebText, and Stories.
- BERT trained for 1M steps, and RoBERTa trained for 500k steps (with substantially larger batch sizes; the net effect being that ==RoBERTa saw many more instances==).
- BERT team thought they should train on short sequences first, in a curriculum learning fashion. RoBERTa trained only on full-length sequences, dropping that.


# Non-Paper FIgures
![[Pasted image 20240619172710.png]]



