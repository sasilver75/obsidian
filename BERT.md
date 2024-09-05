---
aliases:
  - Bidirectional Encoder Representations from Transformers
---
October 11, 2018 -- [[Google Research]]
Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
#zotero 
Takeaway: Using a transformer encoder with bidirectional attention pretrained with a MLM + NSP objective, we can get representations that are then useful for many downstream tasks!

----



Known limitations (From Devlin et al, 2019, and Yang et al, 2019):
- The original BERT paper is admirably detailed, but it's still very partial in its ablation studies, and in studies of how to optimize the model; in the original paper, we're not looking at the best BERT possible.
- We're creating a mismatch between pretraining and finetuning, since the MASK token is never seen during finetuning.
- The downside of using an MLM that's only 15% of tokens are predicted in each batch. We turn off the modeling objective for the tokens that we don't mask, and we only mask a tiny number of them because we need the bidirectional context to make the prediction.
- BERT assumes that the predicted tokens are independent of eachother given the unmasked tokens, which oversimplifies, since high-order, long-range dependencies are prevalent in natural language. If you mask out both NEW and YORK, it's going to be hard to predict either of them. (Yang's ExcelNet brings this back in, to powerful effect.)

Abstract:
> We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, ==BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers==. As a result, the pre-trained BERT model ==can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks==, such as question answering and language inference, without substantial task-specific architecture modifications.


![[Pasted image 20240124182739.png]]

![[Pasted image 20240425144622.png]]

![[Pasted image 20240425191148.png]]
![[Pasted image 20240425191735.png]]
Above: The segment embeddings denote whether a token belongs to "sentence A" or "sentence B", for the Next-Sentence-Prediction objective (later dropped by [[RoBERTa]])

# Non-paper Figures
![[Pasted image 20240619172701.png]]