March 27, 2024 -- [[DataBricks]]
Blog: [Introducing DBRX: A New State-of-the-Art Open LLM](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

A [[Decoder-Only Architecture]] LLM trained with next-token prediction, using an ==[[Mixture-of-Experts|MoE]] architecture== (==132B== total parameters, ==36B active== on any input). Trained on 12T tokens of text and code data with a maximum context length of 32k tokens.

DBRX
DRBX-Instruct

Summary
> Today, we are excited to introduce DBRX, an open, general-purpose LLM created by Databricks. Across a range of standard benchmarks, DBRX sets a new state-of-the-art for established open LLMs. Moreover, it provides the open community and enterprises building their own LLMs with capabilities that were previously limited to closed model APIs; according to our measurements, it ==surpasses GPT-3.5==, and it is competitive with Gemini 1.0 Pro. It is an especially capable code model, surpassing specialized models like CodeLLaMA-70B on programming, in addition to its strength as a general-purpose LLM.
> This state-of-the-art quality comes with marked improvements in training and inference performance. DBRX ==advances the state-of-the-art in efficiency among open models== thanks to its fine-grained ==mixture-of-experts (MoE) architecture==. Inference is up to 2x faster than LLaMA2-70B, and DBRX is about 40% of the size of Grok-1 in terms of both total and active parameter-counts. When hosted on Mosaic AI Model Serving, DBRX can generate text at up to 150 tok/s/user. Our customers will find that training MoEs is also about 2x more FLOP-efficient than training dense models for the same final model quality. End-to-end, our overall recipe for DBRX (including the pretraining data, model architecture, and optimization strategy) can match the quality of our previous-generation MPT models with nearly 4x less compute.
The weights of the base model ([DBRX Base](https://huggingface.co/databricks/dbrx-base)) and the finetuned model ([DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)) are available on Hugging Face under an ==open license==.