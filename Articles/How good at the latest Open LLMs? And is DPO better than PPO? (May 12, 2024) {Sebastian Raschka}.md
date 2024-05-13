#article 
Link: https://magazine.sebastianraschka.com/p/how-good-are-the-latest-open-llms

This is a rundown of recent LLMs by Seb

------

Spring is finally here, and four major open LLM releases:
1. Mistral's [[Mixtral]]
2. Meta's [[LLaMA 3]]
3. Microsoft's [[Phi-3]]
4. Apple's [[OpenELM]]

In addition to those models, we've got some new research on alignment-related methods, from PPO to DPO:

----

# Mixtral, LLaMA 3, and Phi-3: What's new?

### Mistral 8x22B: Larger models are better!
- The [[Mixtral]] 8x22B [[Mixture-of-Experts]](MoE) model by Mistral is similar to the 8x7B Mixtral model released in January 2024.
- In this model, we replace the feed-forward with 8 expert layers (and a router).
![[Pasted image 20240513005435.png|300]]
Above: A claim that shows that Mixtral performance maximizes MMLU performance while minimizing active parameter counts.

### LLaMA 3: Larger data is better!
- Meta's first LLaMA model releases in Feb 2023 were big breakthroughs for openly-available LLMs. LLaMa 2 followed in July 2023, and now we've got the third iteration.
- Meta's still training their large 400B variant of it, leaving us initially with an 8B and 70B pair of variants. Interestingly, all of these models are *dense* models.
- The main differences between LLaMA 2 and 3 is that the latter has an increased vocabulary size, context, and the fact that LLaMA 3 uses [[Grouped Query Attention]]. In addition LLaMA 3 is trained on 15T tokens, as opposed to "only" 2T tokens for LLaMA 2. This is well beyond Chinchilla optimality.
![[Pasted image 20240513005852.png]]
Above: See LLaMA added to the same chart, with both 8B and 70B variants of LLaMA 3.






















