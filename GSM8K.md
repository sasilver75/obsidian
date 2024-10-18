October 27, 2021 -- [[OpenAI]]
Paper: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

Notes: 
- Used by [[LLaMA 3.1]]

Abstract
> State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we introduce ==GSM8K==, a ==dataset of 8.5K high quality linguistically diverse grade school math word problems==. We find that ==even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution==. To increase performance, we propose training verifiers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked highest by the verifier. We demonstrate that verification significantly improves performance on GSM8K, and we provide strong empirical evidence that verification scales more effectively with increased data than a finetuning baseline.




# Paper Figures

# Non Paper Figures

![[Pasted image 20240513150845.png]]
Above: A recent study found that several model families such as Phi and Mistral models show evidence of systematic overfitting on the GSM8k grade school math dataset. Note that one way of overfitting is through the creation of synthetic data which may inadvertently reflect use cases in the test data, rather than a broader set of model applications.

![[Pasted image 20241017223754.png]]