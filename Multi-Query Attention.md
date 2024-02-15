Attention variant used in the [[PaLM]] paper; Unlike in [[Multi-Head Attention]], the key/value projections are shared for each head. This takes the same training time, but faster autoregressive decoding in inference.

Compare with: [[Grouped Query Attention]], which might be a better alternative if you care about improved speed but still want to maintain maximal accuracy.