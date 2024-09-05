February 27, 2023 (3 months after [[ChatGPT]] (GPT 3.5))
[[Meta AI Research]]
Paper: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)("Large Language Model Meta AI")
Followed by: [[LLaMA 2]]
#zotero
Significance: [[Meta AI Research]] is the only company to go into [detail](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)  about their experience training a 175B-parameter model (Meta OPT-175B. Because LLaMA was just a base model, projects like Stanford's [[Alpaca]] went ahead and instruction-tuned it in the following weeks and set off the open-source finetuning wave.

----

Takeaways:
- A collection of LMs ranging from 7B-65B parameters, with LLaMA-13B outperforming [[GPT-3]] on most benchmarks, and LLaMA-65B is competitive with [[Chinchilla]] 70B and [[PaLM]] 540B. The point was to show that you can do this using public datasets exclusively.
- Notes that while [[Chinchilla]] described optimal training given a specific training compute budget, most of what matters is the performance and cost at inference time, so they train well-beyond the Chinchilla frontier.
	- (While Chinchilla would recommend training a 10B model on 200B tokens, they find that the performance of a 7B model continues to improve after even 1T tokens)
- They note that their experience *briefly* finetuning models on instruction-following datasets led to improvements in benchmarks, but they don't really mention much else about it in the paper, saying that "we plan to investigate this in further work."
	- Note that LLaMA is *just a base model*, not a meaningfully instruction-tuned model.

Specs:
- Pretraining dataset includes [[Common Crawl]],[[C4]], Github, Wikipedia, Books, ArXiv, StackExchange.
- Uses [[Byte-Pair Encoding|BPE]], using the implementation from [[SentencePiece]] (Kudo and Richardson, 2018)
- Uses *pre-normalization* (a la GPT-3), normalizing the *input* of each Transformer sublayer, instead of the output.
- [[SwiGLU]] activation functions (a la PaLM)
- [[Rotary Positional Embedding]] (a la GPTNeo) at each layer of the network, instead of absolute positional embeddings.
- Uses the [[AdamW]] optimizer, and a cosine learning rate schedule.


Abstract
> We introduce LLaMA, a ==collection of foundation language models ranging from 7B to 65B parameters==. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, ==LLaMA-13B outperforms GPT-3 (175B) on most benchmarks==, and ==LLaMA-65B is competitive with the best models, [[Chinchilla]]-70B and [[PaLM]]-540B==. We release all our models to the research community.

 Combined a bunch of the best features from [[PaLM]] and [[Chinchilla]]:
	- Pre-normalize the input of each transformer sub-layer
	- Use [[RMSNorm]] instead of [[Layer Normalization|LayerNorm]]
	- [[SwiGLU]] activation function from [[PaLM]]
	- Uses [[Rotary Positional Embedding|RoPE]], like [[PaLM]] did.
	- Uses [[AdamW]], like [[Chinchilla]] did.

![[Pasted image 20240417224708.png]]
![[Pasted image 20240427133917.png|400]]
![[Pasted image 20240427140609.png]]
