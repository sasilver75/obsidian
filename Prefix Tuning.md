January 1, 2021
Stanford University
Paper: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
...
Takeaway: Instead of adding a soft prompt to the model's input (as in the later [[Prompt Tuning]] paper), we prepend trainable parameters to the hidden states of all of our transformer blocks. During fine-tuning, we keep all of the LM's original parameters frozen while the prefix parameters are updated. Paper claimed/showed this achieved similar performance to full fine-tuning despite requiring updates on just 0.1% of parameters, and, in settings involving extrapolation to new topics, it outperformed full fine-tuning; a hypothesis is that training fewer parameters helped reduce overfitting in smaller target datasets.

Related: 3 months later, Google came out with a paper on [[Prompt Tuning]], which they referred to as a "simplification of the recently proposed prefix tuning method."

----

We freeze all the parameters of the pretrained network itself, and never change any of them. Instead, we make a bunch of fake pseudo-word vectors that we prepend to the beginning of a sequence, and we just train them!
- These would have been inputs to the network, but we specify them as parameters, and just train the values/parameters of the fake words.
- This keeps all the generality of the model params, and is easier to do than finetune the entire network.

Abstract
> Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose ==prefix-tuning==, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but ==optimizes a small continuous task-specific vector== (called the prefix). Prefix-tuning draws inspiration from prompting, ==allowing subsequent tokens to attend to this prefix as if it were "virtual tokens".== We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1\% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.

# Paper Figures
![[Pasted image 20240525150355.png]]

# Non-Paper Figures

![[Pasted image 20240410180558.png]]