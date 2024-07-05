June 17, 2024
[[NVIDIA]]
Paper: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704v1)

> "[[Nemotron-4]]'s entire paper is focused on the training of a very good reward model, to then use as their data filterer."

> "Good for synthetic generation because of its permissive use license, compared to models like GPT-4." - Zeta Alpha folks

Abstract
> We release the Nemotron-4 340B model family, including ==Nemotron-4-340B-Base==, ==Nemotron-4-340B-Instruct==, and ==Nemotron-4-340B-Reward==. Our models are open access under the NVIDIA Open Model License Agreement, a permissive model license that allows distribution, modification, and use of the models and its outputs. These models perform competitively to open access models on a wide range of evaluation benchmarks, and were sized to fit on a single DGX H100 with 8 GPUs when deployed in FP8 precision. We believe that the community can benefit from these models in various research studies and commercial applications, especially for generating synthetic data to train smaller language models. ==Notably, over 98% of data used in our model alignment process is synthetically generated==, showcasing the effectiveness of these models in generating synthetic data. To further support open research and facilitate model development, we are also open-sourcing the synthetic data generation pipeline used in our model alignment process.