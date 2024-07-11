Feb 29, 2024 -- A huge assembly of researchers from 38 places, including [[HuggingFace]], [[EleutherAI]], and more
Paper: [StarCoder 2 and The Stack v2: The Next Generation](https://arxiv.org/abs/2402.19173)
HuggingFace Dataset: [The Stack V2](https://huggingface.co/datasets/bigcode/the-stack-v2)

The dataset released along with the [[StarCoder 2]] model; mostly developed by [[EleutherAI]]
900B unique tokens+ un

Abstract
> The BigCode project, an open-scientific collaboration focused on the responsible development of Large Language Models for Code (==Code LLMs==), introduces ==StarCoder2==. In partnership with Software Heritage (SWH), we build ==[[The Stack v2]]== on top of the digital commons of their source code archive. Alongside the SWH repositories spanning ==619 programming languages==, we carefully select other high-quality data sources, such as GitHub pull requests, Kaggle notebooks, and code documentation. This ==results in a training set that is 4x larger than the first StarCoder dataset==. We train StarCoder2 models with ==3B, 7B, and 15B parameters on 3.3 to 4.3 trillion tokens== and thoroughly evaluate them on a comprehensive set of Code LLM benchmarks. We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2- 15B, significantly outperforms other models of comparable size. In addition, it matches or outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder- 33B is the best-performing model at code completion for high-resource languages, we find that StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several low-resource languages. We make the model weights available under an OpenRAIL license and ensure full transparency regarding the training data by releasing the SoftWare Heritage persistent IDentifiers (SWHIDs) of the source code data.

![[Pasted image 20240419194948.png]]


![[Pasted image 20240229094543.png]]
The Stack v2 is the largest open code dataset suitable for LLM pretraining. The Stack v2 is larger than The Stack v1, follows an improved language and license detection procedure, and better filtering heuristics. In addition, the training dataset is grouped by repositories, allowing to train models with repository context.

