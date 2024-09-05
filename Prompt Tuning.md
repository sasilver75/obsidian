April 18, 2021
Google Research
Paper: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
...
Takeaway: ...

Related: This is a slightly different concept than [[Prefix Tuning]] (January 2021, Stanford)
- Our method can be seen as a simplification of the recently proposed "prefix tuning" of Li and Liang (2021)"

---

In Prompt tuning, we prepend a trainable tensor to the model's input embeddings, essentially creating a soft prompt. These new "tokens" can be learned via backpropagation.



![[Pasted image 20240525145737.png]]
These blue vectors are prepend to our input sequence. We can freeze the parameters of the model and train these exclusively.

Abstract
> In this work, we explore "==prompt tuning==", a simple yet effective mechanism for ==learning "soft prompts" to condition frozen language models to perform specific downstream tasks==. Unlike the discrete text prompts used by GPT-3, soft prompts are ==learned through backpropagation== and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's "few-shot" learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method "closes the gap" and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. ==Our method can be seen as a simplification of the recently proposed "prefix tuning" of Li and Liang (2021),== and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning.

