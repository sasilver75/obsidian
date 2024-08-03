July 7, 2021 -- [[OpenAI]]
Paper: [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

OpenAI ==code completion benchmark== of coding tasks and unit tests, introduced with the [[Codex]] model.

Notes:
- Used by [[LLaMA 3.1]]


Abstract
> We introduce Codex, a GPT language model fine-tuned on publicly available code from GitHub, and study its Python code-writing capabilities. A distinct production version of Codex powers GitHub Copilot. On ==HumanEval==, a ==new evaluation set we release to measure functional correctness for synthesizing programs from docstrings==, our model solves 28.8% of the problems, while GPT-3 solves 0% and GPT-J solves 11.4%. Furthermore, we find that repeated sampling from the model is a surprisingly effective strategy for producing working solutions to difficult prompts. Using this method, we solve 70.2% of our problems with 100 samples per problem. Careful investigation of our model reveals its limitations, including difficulty with docstrings describing long chains of operations and with binding operations to variables. Finally, we discuss the potential broader impacts of deploying powerful code generation technologies, covering safety, security, and economics.

![[Pasted image 20240420132146.png]]

![[Pasted image 20240420132232.png]]