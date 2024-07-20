November 25, 2022 (6 months before [[Let's Verify Step by Step]])
[[DeepMind]] (Uesato et al)
[Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
#zotero 
Takeaway: 

Note: A year after this DeepMind paper, OpenAI did their [[Let's Verify Step by Step]] paper, which similarly did an analysis of PRMs applied to math, but the used a more capable base model, significantly more human feedback, and used the more challenging [[MATH]] dataset, instead of [[GSM8K]] (these were all critiques that OpenAI had of this GDM paper).


---

## Introduction
- Recent work has shown language models using step-by-step reasoning (eg) [[Chain of Thought|CoT]] to solve tasks.
- How should we best supervise such models?
	- Outcome-based approaches: Supervise the final result
	- Process-based approaches: Supervise each step of the reasoning process, including the last step outputting the final result.
- Process-based approaches emphasize human understanding -- in order to either demonstrate or select good reasoning steps, human annotators need to understand the task.
- An answer without an understandable explanation may often confuse more than it implies.
	- ((What you cannot explain, you don't understand.))
	- ((Correct answer with incorrect explanation isn't aligned behavior, because in some cases they clearly weren't following that reasoning to get to the correct answers -- they're "lying."))
- Authors conduct a comprehensive comparison between process and outcome-based approaches in the context of [[GSM8K]] math work problems.
	- We vary whether or not supervision is provided only on the final answers (outcome-based) or on individual reasoning steps (process-based).
	- All models used are based on [[Chinchilla]] (70B params, 1.4T tokens)
		- ((This was later a criticism by the [[Let's Verify Step by Step]] paper, but TBH I think this is large enough of a model to benefit from a PRM... It must come down to the scale of the supervision provided?))


## Problem and Methods



## Results



## Discussion



## Related Work




## Conclusion






Abstract
> Recent work has shown that asking language models to generate reasoning steps improves performance on many reasoning tasks. When moving beyond prompting, this raises the question of how we should supervise such models: outcome-based approaches which supervise the final result, or process-based approaches which supervise the reasoning process itself? Differences between these approaches might naturally be expected not just in final-answer errors but also in reasoning errors, which can be difficult to detect and are problematic in many real-world domains such as education. We run the first comprehensive comparison between process- and outcome-based approaches trained on a natural language task, GSM8K. We find that pure outcome-based supervision produces similar final-answer error rates with less label supervision. However, for correct reasoning steps we find it necessary to use process-based supervision or supervision from learned reward models that emulate process-based feedback. In total, we improve the previous best results from 16.8% → 12.7% final-answer error and 14.0% → 3.4% reasoning error among final-answer-correct solutions.


# Paper Figures