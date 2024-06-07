
EvalGen focuses on teaching users the process of creating domain-specific evals by deeply involving the user each step of the way, from specifying criteria, to labeling data, to checking evals. It guides the user through a best practice of crafting LLM evaluations, namely:

1. Defining domain-specific tests (bootstrapped automatically from the prompt). These are defined as either assertions with code or with LLM-as-a-Judge.
2. The importance of aligning the tests with human judgment, so that the user can check that the tests capture the specified criteria.
3. Iterating on your tests as the system (prompts, etc.) changes. 

EvalGen provides developers with a mental model of the evaluation building process without anchoring them to a specific tool. We have found that after providing AI engineers with this context, they often decide to select leaner tools or build their own.

![[Pasted image 20240606165212.png]]