---
aliases:
  - Gated Linear Unit
---
Dec 23, 2016
[[Meta AI Research]]
[Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)

---

An [[Activation Function]] used in various transformer architectures, and notably part of the popular [[SwiGLU]] activation function.  

$GLU(a,b) = a \odot \sigma(b)$ , where $\sigma$ is the [[Sigmoid Activation Function|Sigmoid]] and $\odot$ is the [[Hadamard Product]].
- Here, $a$ and $b$ are often an input $x$ going through two separate linear transformations (eg $a=xW_1$ and $b=xW_2$ ).
	- One goes through a sigmoid activation (the "gate"), and the other remains linear; these two paths are then element-wise multiplied.

Often used in natural language processing, where $b$ is the gate that controls which information from $a$ is passed into the following layer (allowing selection of words or features useful for predicting the next word).

Allows the network to control information flow, learning to select or suppress certain input features. Also helps mitigate vanishing gradient problems and leads to faster convergence in training (?).

Abstract
> The pre-dominant approach to language modeling to date is based on recurrent neural networks. Their success on this task is often linked to their ability to capture unbounded context. In this paper we develop a finite context approach through stacked convolutions, which can be more efficient since they allow parallelization over sequential tokens. We propose a novel simplified gating mechanism that outperforms Oord et al (2016) and investigate the impact of key architectural decisions. The proposed approach achieves state-of-the-art on the WikiText-103 benchmark, even though it features long-term dependencies, as well as competitive results on the Google Billion Words benchmark. Our model reduces the latency to score a sentence by an order of magnitude compared to a recurrent baseline. To our knowledge, this is the first time a non-recurrent approach is competitive with strong recurrent models on these large scale language tasks.