November 28, 2032
Paper: [The Falcon Series of Open Language Models](https://arxiv.org/abs/2311.16867)
From the UAE's Technology Innovation Institute (TII)
Trained using the [[RefinedWeb]] corpus

Abstract
> We introduce the Falcon series: ==7B, 40B, and 180B parameters causal decoder-only models== trained on a diverse high-quality corpora predominantly assembled from web data. The largest model, ==Falcon-180B==, has been trained on over ==3.5 trillion tokens== of text--the largest openly documented pretraining run. Falcon-180B significantly ==outperforms models such as PaLM or Chinchilla, and improves upon concurrently developed models such as LLaMA 2== or Inflection-1. It ==nears the performance of PaLM-2-Large== at a reduced pretraining and inference cost, making it, to our knowledge, one of the three best language models in the world along with GPT-4 and PaLM-2-Large. We report detailed evaluations, as well as a deep dive into the methods and custom tooling employed to pretrain Falcon. Notably, we report on our custom distributed training codebase, allowing us to efficiently pretrain these models on up to 4,096 A100s on cloud AWS infrastructure with limited interconnect. We release a 600B tokens extract of our web dataset, as well as the Falcon-7/40/180B models under a permissive license to foster open-science and accelerate the development of an open ecosystem of large language models.


![[Pasted image 20240417224708.png]]