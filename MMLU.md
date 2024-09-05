---
aliases:
  - Massive Multitask Language Understanding
---
September 7, 2020 -- Primarily UC Berkeley (minor: Columbia, UIUC, UChicago)
Paper: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

Leaderboard: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu

---

A set of 57 tasks that span elementary math, US history, computer science, law, and more. To perform well, models must possess extensive world knowledge and problem-solving ability.

Dwarkesh "*Will Scaling Work*" Skeptic: "have you actually tried LOOKING at a random sample of questions from [[MMLU|MMLU]] and [[BIG-Bench]]? They're almost *all* just Google Search first hit results. They're good tests of *memorization*, not *intelligence!*"

Note: In May 2024, researchers released [[MMLU-Pro]], which increases the MC number of questions from 4-10, and incorporates 12187 hard questions from Scibench, TheoremQA, STEM sites, while also subslicing the original MMLU to remove trivial and ambiguous questions.

> Swyx: "I think engineers need to be thinking about what's next after MMLU. I have 10 proposals, half come from the HuggingFace episode with Clementine Fourier. [[MuSR]] for multistep reasoning, [[MATH]] for math, [[IFEval]] for instruction following, [[BIG-Bench Hard|BBH]], and for Code... I'm considering making my own because I care so much. [[MBPP]] is sort of a replacement for [[HumanEval]], since HE is widely known to be saturated. and [[SciCode]] is a new one. Context utilization has [[ZeroSCROLLS]] and [[InfiniteBench]] (used by L3.1). For tool use, [[Berkeley Function-Calling Leaderboard|BFCL]] and [[Nexus]] are pretty consensus. For multilinguality, we have [[MGSM]]"  [link](https://www.latent.space/p/q2-2024-recap)


Abstract
> We propose a new test to ==measure a text model's multitask accuracy==. The test covers ==57 tasks== including ==elementary mathematics, US history, computer science, law, and more==. To attain high accuracy on this test, ==models must possess extensive world knowledge and problem solving ability==. We find that while most recent models have near random-chance accuracy, the very largest GPT-3 model improves over random chance by almost 20 percentage points on average. However, on every one of the 57 tasks, the best models still need substantial improvements before they can reach expert-level accuracy. Models also have lopsided performance and frequently do not know when they are wrong. Worse, they still have near-random accuracy on some socially important subjects such as morality and law. By comprehensively evaluating the breadth and depth of a model's academic and professional understanding, our test can be used to analyze models across many tasks and to identify important shortcomings.

![[Pasted image 20240420131953.png]]

![[Pasted image 20240420131701.png]]

![[Pasted image 20240420131710.png]]