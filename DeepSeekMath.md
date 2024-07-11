Feb 5, 2024 (One month after [[DeepSeek-Coder]]) -- [[DeepSeek]]
Paper: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

A continued finetune of the existing [[DeepSeek-Coder]], training on 120B math-related tokens from [[Common Crawl]] (along with some natural language and code data).
- They highlight a "meticulously engineering data selection pipeline", which might be interesting to learn about.
- Introduces [[Group Relative Policy }Optimization]] (GRPO), a variant of [[Proximal Policy Optimization]]

Abstract
> Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce ==DeepSeekMath 7B==, which ==continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl==, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, ==approaching the performance level of Gemini-Ultra and GPT-4==. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a ==meticulously engineered data selection pipeline==. Second, we introduce ==Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO)==, that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

