July 26, 2019
Paper: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) ("Robust" BERT)
Compare: [[Bidirectional Encoder Representations from Transformers|BERT]]

Abstract:
> Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes, and, as we will show, ==hyperparameter choices have significant impact on the final results==. ==We present a replication study of BERT pretraining== (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and training data size. ==We find that BERT was significantly undertrained==, and can match or exceed the performance of every model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results ==highlight the importance of previously overlooked design choices, and raise questions about the source of recently reported improvements==. We release our models and code.

((Turns out we just scale the models bigger and nail the hyperparameters and we get a great BERT model!))





