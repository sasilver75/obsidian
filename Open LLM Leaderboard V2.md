June 26, 2024
[[HuggingFace]]
Blogpost: [Performances are plateauing, let's make the leaderboard steep again](https://huggingface.co/spaces/open-llm-leaderboard/blog?utm_source=ainews&utm_medium=email&utm_campaign=ainews-et-tu-mmlu-pro)

Blogpost Notes
- Evaluating and comparing LLMs is hard; papers use optimized prompts and evaluation setups to give the best chance to their model, and often don't give reproducible code.
- The initial Open LLM Leaderboard was a place where reference models could b evaluated in the same setup (same questions, asked in the same order, etc.) to gather reproducible and comparable results -- it's powered by [[Eleuther LM Evaluation Harness]].
- People use OpenLLM leaderboard both to:
	- Find trustable SoTA open-source releases to use for their projects.
	- Evaluate their work (eg pretraining or finetuning) by comparing it to the best existing open models.
- ![[Pasted image 20240709175835.png]]
- Above:
	1. Existing benchmarks became too easy for models (saturation of benchmarks)
	2. Some newer models showed signs of contamination (meaning the models were possibly trained on benchmark data, or very similar data... or even were *derived* (eg via distillation, merging) from models that were contaminated), so the scores don't reflect the general performance of the model.
	3. Some benchmarks contained errors -- MMLU in particular was modified to be [[MMLU-Redux]] and [[MMLU-Pro]]. [[GSM8K]] used a specific end-of-generation token (:), which unfairly pushed down the performance of many verbose models.
- So HuggingFace wanted to find new benchmarks with uncontaminated, high-quality datasets, using reliable metrics and measuring capabilities of interest:
	- Knowledge-testing 📚
	- Reasoning on short and long contexts 💭
	- Complex mathematical abilities 🤓
	- Tasks well-correlated with human preference, like instruction-following 🤝

To that end, Open LLM Leaderboard V2 covers these tasks using ==six benchmarks==:
- 📚[[MMLU-Pro]] (Massive Multitask Language Understanding - Pro): A refined version of the MMLU dataset, which has been the reference multi-choice knowledge dataset.
	- Some questions in the original MMLU are noisy (unanswerable) and now too easy (for stronger models and increased contamination).
	- MMLU-Prof has 10 choices instead of 4 and has been expertly reviewed to reduce the amount of noise, so it's higher-quality and harder than MMLU.
- 📚[[GPQA]] (Google-Proof Q&A Benchmark): An extremely hard knowledge dataset, where questions are designed and reviewed by domain experts in their field (PhD-level biology, physics, chemistry) to be hard to answer by laypeople but (relatively) easy for experts. The dataset is only accessible through gating mechanisms, which should reduce contamination risks.
- 💭[[MuSR]] (Multistep Soft Reasoning): A dataset of algorithmically-generated complex problems of around 1k words in length; the problems are murder mysteries, object placement questions, or team allocation optimizations. Models need to combine reasoning and very long-range context parsing - few models score better than random.
-  🤓[[MATH]] (Mathematics Aptitude Test of Heuristics, using the Level 5 subset): A compilation of high-school-level competition problems gathered from several sources, formatted consistently using Latex for equations and Asymptote for figures, and filtered to keep the hardest questions. Generations must fit a very specific output format.
- 🤝[[IFEval]] (Instruction-Following Eval): Tests the capabilities of models to clearly follow explicit instructions, like "include keyword x" or "use format y." The models are tested on their ability to strictly follow formatting instructions rather than the actual contents generated, allowing strict and rigorous metrics to be used.
- 🤓🤝[[BIG-Bench Hard|BBH]] (BIG-Bench Hard): A subset of 23 challenging tasks from the [[BIG-Bench]] dataset (204 tasks), which use objective metrics, are hard, and contain enough samples to be statistically significant. Contain multistep arithmetic and algorithmic reasoning (understanding boolean expressions, SVG for geometric shapes, etc), language understanding (sarcasm detection, name disambiguation, etc) and some world knowledge.

Collectively, these benchmarks and subsets were chosen because of:
1. Evaluation quality (human reviewed and widespread use)
2. Reliability of fairness of metrics (multi-choice evaluations are fair across models; generative evaluations should constrain the format like MATH or use unambiguous metrics like IFEval or post-processing like BBH to extract correct answers)
3. General absence of contamination in today's models

==How model rankings are calculated==: Instead of summing each benchmark score as our composite score, we normalize single-test scores between the random-choice baseline and the maximal possible score (100 points). We then average all of the normalized scores to get the final average score with which to compute final rankings.
- This basically changes the weight assigned to each benchmark in the final average score.
![[Pasted image 20240709184543.png|400]]
If you looked at raw scores, you would assume that MATH Level 5 and MMLU-Pro are the hardest benchmarks, but with normalization, the hardest seem to be MuSR and MATH Level 5.

They had been using an older pinned version of Eleuther Evaluation Harness for consistency of evaluation. As time went on (and the Harness versions progressed), people were confused why they would get different results than the Open LLM Leaderboard. So they've worked with the EleutherAI team to update the harness, adding support for ==delta weights== ([[Low-Rank Adaptation|LoRA]] finetuning/adaptation of models), a logging system compatible with the leaderboard, and the highly-requested use of chat templates for evaluation.

To highlight high-quality models in the leaderboard and prioritize the most useful models for evaluation, HF has introduced a category called "maintainer's choice," which are models handpicked by the HF team (from hyperscalers, industry labs, open research groups, and even users who have shipped great models).

In the original OpenLLM leaderboard, evaluations run in a FIFO queue; now, there's a voting system for submitted models, where models voted for by the community will be prioritized first, surfacing the most awaited models at the top of the priority queue.

Interesting results of the V1 -> V2 swap on leaderboard rankings
- HF very impressed by [[Qwen 2]]-72B-Instruct scoring at the top of the leaderboard.
- LLaMA-3-70B-Instruct interestingly loses 15 points to its pretrained basemodel counterpart on GPQA, begging the question of whether extensive instruction fine-tuning done by the Meta team on the model affected some expert/graduate-level knowledge ([[Alignment Tax]]).

Which of the six evaluations should you pay attention to?
![[Pasted image 20240709191843.png]]
- See that MMLU-Pro and BBH are well-correlated (and these benchmarks are also quite correlated with human preference).
- IFEval targets chat capabilities, but the format used in the benchmark tends to favor chat and instruction-tuned models, with pretrained models having a harder time reaching higher performances.

![[Pasted image 20240709192036.png]]
