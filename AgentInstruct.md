---
aliases:
  - Orca 3
---
July 3, 2024
[[Microsoft Research]]
[AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502v1)
#zotero 
Takeaway: ...


---


Abstract
> Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns around ==model collapse== and ==drawbacks of imitating other models==. This discrepancy can be attributed to the fact that synthetic data varies in quality and diversity. ==Effective use of synthetic data usually requires significant human effort in curating the data==. We focus on using synthetic data for post-training, specifically creating data by powerful models to teach a new skill or behavior to another model, we refer to this setting as ==Generative Teaching==. We introduce ==AgentInstruct==, an ==extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data==. AgentInstruct can create both the prompts and responses, using only raw data sources like text documents and code files as seeds. We demonstrate the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills, such as text editing, creative writing, tool usage, coding, reading comprehension, etc. The dataset can be used for instruction tuning of any base model. We post-train Mistral-7b with the data. When comparing the resulting model Orca-3 to Mistral-7b-Instruct (which uses the same base model), we observe significant improvements across many benchmarks. For example, 40% improvement on AGIEval, 19% improvement on MMLU, 54% improvement on GSM8K, 38% improvement on BBH and 45% improvement on AlpacaEval. Additionally, it consistently outperforms other models such as LLAMA-8B-instruct and GPT-3.5-turbo.