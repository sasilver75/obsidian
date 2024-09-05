This is just in contrast to unidirectional attention (as seen in [[Masked Attention]]/[[Masked Attention|Causal Attention]]); In this scheme, an element can attend to every element in the sequence, regardless of whether that elements comes "before" or "after" it in a sequence. 

Transformer Encoders (eg like those used in [[BERT|BERT]]) use bidirectional/full attention.
In contrast, Decoder self-attention is unidirectional and masked, and Decoder cross-attention to the encoder is similarly unidirectional (in the sense that each decoder token attends to the entire decoder output).