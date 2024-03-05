---
aliases:
  - Causal Attention
---


(Basically, in a normal self-attention, each token computes attention weights with every other token in the sequence. In a causally masked self-attention, tokens do not compute attentions with tokens that come after themselves)
(Generally, a masked attention just means that there are certain tokens that a token can't attend to. In causal attention/causally-masked attention of tokens in a sequence, that means that tokens can't attend to future tokens. It seems that causal attention/causally-masked attention is a subset of masked attention generally)
(The purpose of the triangular causal mask in the attention mechanism is to enforce the autoregressive property of the model during training and inference.)