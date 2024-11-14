---
aliases:
  - Latent Langauge Image Pretraining
---
Lavoie et al. 2024
An improvement on [[CLIP]] and [[SigLIP]] which takes advantage of the idea that a caption can be captioned in several different ways. Proposes to condition the encoding of an image on the target caption via cross-attention module. 
Accounting for caption diversity increases the representation's expressivity and it generally improves downstream zero-shot transfer classification and retrieval performance.

