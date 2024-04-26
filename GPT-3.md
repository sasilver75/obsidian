May 28, 2020
Paper: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Referring to the model's capability to do [[In-Context Learning]])

175B parameter autoregressive language model trained on ~300B tokens.
((In retrospect, wildly overparametrized for the amount of data it was trained on))

Abstract:
> Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train ==GPT-3==, an ==autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model==, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 ==achieves strong performance on many NLP datasets==, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

Variants:
- text-davinci-001 (June 2020; 13B)
- text-davinci-002 (*July 2021*; 6B)
- text-davinci-003 (June 2020; 175B)
- text-ada-001 (June 2020; .125B)
- text-babbage-001 (June 2020; 1.2B)
- text-curie-001 (June 2020; 6.7B)
- davinci-instruct-beta (May 2021; 2.7B instruction-tuned)
- davinci (June 2020; 13B, original model)
- curie-instruct-beta (May 2021; 92M)
- curie (June 2020; 2.7B)
- babbage (June 2020; 125M)
- ada (June 2020; 40M)
- Codex/code-davinci-002 (August 2021; 14.8B, GPT-3 fine-tuned on code)

![[Pasted image 20240425141916.png]]