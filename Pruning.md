---
aliases: []
---


Resources:
- Article: [WandB Diving into Model Pruning](https://wandb.ai/authors/pruning/reports/Diving-Into-Model-Pruning-in-Deep-Learning--VmlldzoxMzcyMDg)
- Note: [[Efficient ML (3) - Pruning and Sparsity Part 1]]


The practice of discarding weights in a model that don't improve a model's performance. In this context, we really mean *zero-ing* out non-significant weights.
Careful pruning lets us compress our models so that inference runs faster and so that they can be deployed into (eg) mobile phones and other resource-constrained devices.

The specific patterns (ranging from unstructured to structured) of pruning that we choose to use are very important with respect to hardware acceleration (highly structured/regular pruning is easier for hardware acceleration to take advantage of):
![[Pasted image 20240628125524.png|400]]

Note that [[Quantization]] and [[Distillation]] are orthogonal concepts; We can quantize a pruned matrix, or prune a quantized matrix. We can also prune or quantize a low-rank approximation of a matrix, or vice versa.

## Selection Criteria for Pruning
[[Magnitude-Based Pruning]]: The idea is that weights with small magnitudes are considered less significant, and those with weights below some threshold are pruned. 

[[Scaling-Based Pruning]]: 