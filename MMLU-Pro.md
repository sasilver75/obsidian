May 2024
Note: In May 2024, researchers improved [[MMLU|MMLU]] with [[MMLU-Pro]], which increases the MC number of questions from 4-10, and incorporates 12187 hard questions from Scibench, TheoremQA, STEM sites, while also subslicing the original MMLU to remove trivial and ambiguous questions. It is a *refined* and *expanded* version of MMLU.

Claude 3 Sonnet saw its performance fall from .815 on MMLU to .6793 on MMLU Pro

HuggingFace has already anointed MMLU-Pro as the successor in the [[Open LLM Leaderboard V2]]

Note June 8 ,2024: Some users on r/LocalLLaMA have dug into it and found some issues with [math heaviness](https://www.reddit.com/r/LocalLLaMA/comments/1du52gf/mmlupro_is_a_math_benchmark/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-et-tu-mmlu-pro) ("MMLU-Prof is a benchmark dominated by math, whereas the original MMLU was more of a knowledge and reasoning test, formulated to be mostly answerable without needing CoT"), and later users found discrepancies in how models are evaluated across sampling patterns, system prompts, and answer-extraction regex. The MMLU-Pro team have acknowledged the discrepancies but claim that this has minimal impact (despite the team having given extra attention to the closed models). Disappointing but fixable!
![[Pasted image 20240708201816.png]]