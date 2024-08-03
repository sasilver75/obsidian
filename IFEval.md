---
aliases:
  - Instruction-Following Eval
---
November 14, 2023
Google, Yale
Paper: [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911)

Tests the capabilities of models to clearly follow explicit instructions, like "include keyword x" or "use format y." The models are tested on their ability to strictly follow formatting instructions rather than the actual contents generated, allowing strict and rigorous metrics to be used.

IFEval targets chat capabilities, but the format used in the benchmark tends to favor chat and instruction-tuned models, with pretrained models having a harder time reaching higher performances.

> "IFEval is nice. Lots of post-training teams picking it up -- we're playing with it."
> - Nathan Lambert, June 9 2024
> "IFEval is the only public evaluation that Apple highlighted for Apple Intelligence; the rest were internal benchmarks."

Notes:
- Used by [[LLaMA 3.1]], [[Apple Intelligence Foundation Language Models|Apple Foundation Model]]

Abstract
> One core capability of Large Language Models (LLMs) is to follow natural language instructions. However, the evaluation of such abilities is not standardized: Human evaluations are expensive, slow, and not objectively reproducible, while LLM-based auto-evaluation is potentially biased or limited by the ability of the evaluator LLM. To overcome these issues, we introduce Instruction-Following Eval (==IFEval==) for large language models. IFEval is a straightforward and easy-to-reproduce evaluation benchmark. It ==focuses on a set of "***verifiable instructions***" such as "write in more than 400 words" and "mention the keyword of AI at least 3 times".== We ==identified 25 types of those verifiable instructions and constructed around 500 prompts, with each prompt containing one or more verifiable instructions==. We show evaluation results of two widely available LLMs on the market. Our code and data can be found at [this https URL](https://github.com/google-research/google-research/tree/master/instruction_following_eval)



# Non-Paper Figures
![[Pasted image 20240709192005.png]]
From the [Open LLM Leaderboard V2 blogpost by HF](https://huggingface.co/spaces/open-llm-leaderboard/blog?utm_source=ainews&utm_medium=email&utm_campaign=ainews-et-tu-mmlu-pro)
