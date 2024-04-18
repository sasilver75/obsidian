June 29, 2022 -- Google Research, Blueshift Team
Paper: [Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/abs/2206.14858)

Minerva builds on [[PaLM]], with further training on a 118GB dataset of scientific papers from arXiv, and webpages containing LaTeX/MathJax, etc. By maintaining this symbolic formatting in the data, the model learns to converse by using standard mathematical notation.

Minerva also incorporates [[Chain of Thought]] to solve problems. Rather than generating a single solution, Minerva generates multiple solutions and uses majority voting on these solutions.

Abstract
> Language models have achieved remarkable performance on a wide range of tasks that require natural language understanding. Nevertheless, state-of-the-art ==models have generally struggled with tasks that require quantitative reasoning==, such as solving mathematics, science, and engineering problems at the college level. To help close this gap, we introduce ==Minerva==, a large ==language model pretrained on general natural language data and further trained on technical content==. The model ==achieves state-of-the-art performance on technical benchmarks without the use of external tools==. We also evaluate our model on over two hundred undergraduate-level problems in physics, biology, chemistry, economics, and other sciences that require quantitative reasoning, and find that the model can correctly answer nearly a third of them.

