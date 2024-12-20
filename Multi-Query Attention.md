
References:
- - Video: [Turns out Attention wasn't all we needed - How have modern Transformer architectures evolved?](https://youtu.be/mVLO9PHFc0I?si=rilWUkZy9z8zoFHq)

Attention variant used in the [[PaLM]] paper; Unlike in [[Multi-Headed Attention]], the key/value projections are shared for each head. This takes the same training time, but faster autoregressive decoding in inference.

Compare with: [[Grouped Query Attention]], which might be a better alternative if you care about improved speed but still want to maintain maximal accuracy.