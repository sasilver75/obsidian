Resources:
- [WandB Diving into Model Pruning](https://wandb.ai/authors/pruning/reports/Diving-Into-Model-Pruning-in-Deep-Learning--VmlldzoxMzcyMDg)
- 

The practice of discarding weights in a model that don't improve a model's performance. In this context, we really mean *zero-ing* out non-significant weights.
Careful pruning lets us compress our models so that inference runs faster and so that they can be deployed into (eg) mobile phones and other resource-constrained devices.

**Magnitude-based pruning**: The idea is that weights with small magnitudes are considered less significant, and those with weights below some threshold are pruned. 