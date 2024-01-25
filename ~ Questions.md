- (01/05/2024): So for multi-headed attention, the idea is that each attention head attends to some different aspect of the input, right? What makes that the case? Is it ever the case that multiple attention heads basically attend to the same thing?
- (01/24/2024): The original transformer paper introduced an Encoder-and-Decoder architecture. Later, BERT did the Encoder-Only and GPT did the Decoder-Only. These are respectively good (I think) for representation learning/predictive modeling and text generation. Why didn't they both just use the combined encoder-decoder, like the later BART paper did? What do you gain by just using one of them?



