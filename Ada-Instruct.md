October 6, 2023 -- Shanghai University of Finance and Economics
[Ada-Instruct: Adapting Instruction Generators for Complex Reasoning](https://arxiv.org/abs/2310.04484)

An instruction-generating model created by fine-tuning open-source LLMs.
Referenced as an inspiration in the [[Genstruct]] paper from Nous Research.

Abstract
> ==Generating diverse and sophisticated instructions== for downstream tasks by Large Language Models (LLMs) ==is pivotal== for advancing the effect. Current approaches leverage closed-source LLMs, employing in-context prompting for instruction generation. However, in this paper, ==we found that in-context prompting cannot generate complex instructions with length ≥100 for tasks like code completion==.  
> To solve this problem, we introduce ==Ada-Instruct==, an ==adaptive instruction generator developed by fine-tuning open-source LLMs==. Our pivotal finding illustrates that fine-tuning open-source LLMs with a mere ten samples generates ==long instructions that maintain distributional consistency== for complex reasoning tasks. We empirically validated Ada-Instruct's efficacy across different applications, including code completion, mathematical reasoning, and commonsense reasoning. The results underscore Ada-Instruct's superiority, evidencing its improvements over its base models, current self-instruct methods, and other state-of-the-art models.