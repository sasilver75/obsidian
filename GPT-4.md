March 15, 2023
[[OpenAI]]
[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

> Likely a 1.8T parameter MoE model with 16 experts, each with 111B parameters. -NVIDIA Demo
> "iirc lucas kaiser said in a webinar that GPT4 is trained on 13T tokens" - ChyGao, Interconnects Discord
> " geohot said GPT-4 was 10T tokens, I think dylan (patel) reported something similar as well. So this number seems to be accurate" - Xeophon, Interconnects Discord


Abstract
> We report the development of GPT-4, a large-scale, ==multimodal== model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, ==GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers==. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The ==post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior==. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.