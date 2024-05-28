Link: https://blog.reachsumit.com/posts/2023/03/two-tower-model/

---

## Introduction
- Two-tower models ([[Bi-Encoder]]s) are widely-adopted in industrial-scale retrieval and ranking workflows -- they're the current go-to state-of-the-art solution for pre-ranking tasks.
	- This article explores the history and current-state of Bi-Encoders and highlights potential improvements.

## Cascade Ranking System
- In large-scale information retrieval, search queries might have matching keywords over millions of documents; designing such systems often have to deal with the additional challenges of strict search latency constraints -- even a 100ms increase leads to degraded user experience.
- Because a single, complex ranking algorithm cannot rank such large sets of candidates, a multi-stage ranking system is commonly adapted to balance efficiency and effectiveness.
- ![[Pasted image 20240527211036.png|100]]
- Above: In systems using ==Cascade Ranking Systems==, early on, we use simpler and faster algorithms that focus on [[Recall]] metrics. Later on in the pipeline (on fewer candidates), we use larger-scale NNs for ranking and re-ranking that are higher-latency but more performant (there might be multiple of such steps).

## Two Tower Model
- The ==pre-ranking stage== does initial filtering of candidates retrieved during the preceding recall or ==retrieval== stage. Compared to the ==ranking== stage, *pre-ranking* models pre-ranking models have to evaluate larger numbers of candidates, and have to have a faster approach.![[Pasted image 20240527211529.png]]
- Above: Some examples of generations of pre-ranking systems.
- ![[Pasted image 20240527211712.png]]
- The reason why the two tower rose to popularity was its accuracy as well as its inference efficiency-focused design. The two towers generate latent representations independently in parallel, and interact only at the output layer. Often document embeddings are determined and indexed in something like [[FAISS]] for quicker operations at inference time.

## Related DNN Paradigms
![[Pasted image 20240527212101.png]]
- Above:
	- Figure A is the Two Tower model, a *representation-based* ranker architecture, which independently computes embeddings for the query and documents, and estimates their similarity via interaction at the output layer.
	- Figure B shows models that model word- and phrase-level relationships across query and document using an interaction matrix and then feed it to a neural network like CNN or MLP.
	- Figure C describes Cross-Encoder models are much more powerful, as they model interactions between words across the query and documents at the same time.
	- Figure D: Models like ColBERT keep interactions within the query and document featuring while delaying the query-document interaction to the output layer. This allows the model to preserve the "query-document decoupling" paradigm (which lets us embed and index document embeddings ahead of time).

## Comparing Dual-Encoder Architectures
- Dual encoder or [[Bi-Encoder]] architectures encode the input (such as a query, an image, or a document) into a single dense vector embedding -- Queries receive the same treatment, and the model is optimized based on similarity metrics in embedding space. ![[Pasted image 20240527213216.png]]

## Enhancing the Two Tower Model
- A ==common problem with Bi-Encoders== is the lack of interaction between the two towers. As we saw earlier, the two tower models train the latent embeddings in both towers *independently*, without using an enriching information from the other tower tower.

### Dual Augmented Two-Tower Model (DAT)
- Yu et al proposed augmenting the embedding input of each tower with a vector that captures historical positive interaction information from the other tower.
- These $a_u$ and $a_v$ vectors get updated during the training process and are used to model the information interaction between the two towers by regarding them as the inpnut feature of the two towers.
- ![[Pasted image 20240527213857.png]]
- Unfortunately, later research showed that the gains achieved by the DAT model are still limited

### Interaction Enhanced Two Tower Model (IntTower)
- Authors design a two-tower model that emphasizes both information interactions and inference efficiency. The model has three blocks:
1. Light-SE Block: Used to identify the importance of different features and obtain refined feature representations in each tower. The design of this module is based on the SENET model from computer vision's "Squeeze-and-Excitation Networks" paper.
2. FE-Block: Inspired by the later-interaction style of ColBERT, it performs fine-grained early feature interaction between multi-layer user representations and the last layer of item representation.
3. CIR Module: A Contrastive Interaction Regularization (CIR) module was proposed to shorten the distance between a user and positive items using [[InfoNCE]] loss function. During training, the loss value is combined with the log loss between model prediction scores and true labels.

![[Pasted image 20240527214200.png]]
![[Pasted image 20240527214154.png]]


