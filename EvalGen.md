April 18, 2024
UC Berkeley ([[Shreya Shankar]] et al)
[Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences](https://arxiv.org/abs/2404.12272)
(To read the paper, still)

EvalGen focuses on teaching users the process of creating domain-specific evals by deeply involving the user each step of the way, from specifying criteria, to labeling data, to checking evals. It guides the user through a best practice of crafting LLM evaluations, namely:

1. Defining domain-specific tests (bootstrapped automatically from the prompt). These are defined as either assertions with code or with LLM-as-a-Judge.
2. The importance of aligning the tests with human judgment, so that the user can check that the tests capture the specified criteria.
3. Iterating on your tests as the system (prompts, etc.) changes. 

EvalGen provides developers with a mental model of the evaluation building process without anchoring them to a specific tool. We have found that after providing AI engineers with this context, they often decide to select leaner tools or build their own.

![[Pasted image 20240606165212.png]]

Abstract
> Due to the cumbersome nature of human evaluation and limitations of code-based evaluation, Large Language Models (LLMs) are increasingly being used to assist humans in evaluating LLM outputs. Yet LLM-generated evaluators simply inherit all the problems of the LLMs they evaluate, requiring further human validation. We present a mixed-initiative approach to ``validate the validators'' -- aligning LLM-generated evaluation functions (be it prompts or code) with human requirements. Our interface, EvalGen, provides automated assistance to users in generating evaluation criteria and implementing assertions. While generating candidate implementations (Python functions, LLM grader prompts), EvalGen asks humans to grade a subset of LLM outputs; this feedback is used to select implementations that better align with user grades. A qualitative study finds overall support for EvalGen but underscores the subjectivity and iterative process of alignment. In particular, we identify a phenomenon we dub \emph{criteria drift}: users need criteria to grade outputs, but grading outputs helps users define criteria. What is more, some criteria appears \emph{dependent} on the specific LLM outputs observed (rather than independent criteria that can be defined \emph{a priori}), raising serious questions for approaches that assume the independence of evaluation from observation of model outputs. We present our interface and implementation details, a comparison of our algorithm with a baseline approach, and implications for the design of future LLM evaluation assistants.