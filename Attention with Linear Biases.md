---
aliases:
  - ALiBi
---
A long context positional embedding scheme
Compare with: [[Rotary Positional Embedding|RoPE]]



# Paper Figures
![[Pasted image 20240604191613.png]]
X Axis is how long the token sequences are at test time. All are trained on max-512-token sequences, and we're seeing how good they are at predicting the next word when the sequence length is (eg) 4000. Interesting that this seems to indicate that [[Rotary Positional Embedding|RoPE]] doesn't generalize to longer sequences (than what it's trained on) as well as ALiBi does. 
- Note that usually as you give a model more information, you'd expect the model perplexity to decrease, because it has more context. So ideally we'd hope to see a curve going slightly down!

# Non-Paper Figures
![[Pasted image 20240604190907.png]]