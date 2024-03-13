- (01/05/2024): So for multi-headed attention, the idea is that each attention head attends to some different aspect of the input, right? What makes that the case? Is it ever the case that multiple attention heads basically attend to the same thing? 
- (01/24/2024): The original transformer paper introduced an Encoder-and-Decoder architecture. Later, BERT did the Encoder-Only and GPT did the Decoder-Only. These are respectively good (I think) for representation learning/predictive modeling and text generation. Why didn't they both just use the combined encoder-decoder, like the later BART paper did? What do you gain by just using one of them?
- (1/24/2024):  How does "Curriculum learning" differ from "Active learning" and "Student-teacher" learning (This last one is one that I have in a concept in my head, but it might go by a different name.
- (1/31/2024): When would someone use Rotary Positional Embeddings ([[Rotary Positional Embedding|RoPE]]) vs [[Attention with Linear Biases]]?
- (02/21/2024): Difference between Bi-Encoders vs Cross-Encoders? 
- (02/21/2024): What is "weight tying" as described [here](https://cameronrwolfe.substack.com/p/dolma-olmo-and-the-future-of-open?utm_source=post-email-title&publication_id=1092659&post_id=141461162&utm_campaign=email-post-title&isFreemail=true&r=764e6&utm_medium=email)
- (02/29/2024): Why do we use binary cross entropy for multi-label classification problems, and cross-entropy for single-label classification problems?
- (03/05/2024): Why do logarithms help us "turn products into sums?" log(ab) = log(a) + log(b)
- (03/13/2024): Eugene Cheah mentioned in LLM club that the Falcon paper showed that training on multilingual actually ended up lowering performance on english benchmarks... and that's why many datasets/models are english-only! Is this true? I thought it was the opposite!



