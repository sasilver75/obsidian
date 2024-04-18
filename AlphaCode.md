February 8, 2022 -- [[DeepMind]]
Paper: [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)

A code-generating LLM from [[DeepMind]]

Abstract
> Programming is a powerful and ubiquitous problem-solving tool. Developing systems that can assist programmers or even generate programs independently could make programming more productive and accessible, yet so far incorporating innovations in AI has proven challenging. ==Recent large-scale language models have demonstrated an impressive ability to generate code==, and are now able to complete simple programming tasks. However, these models ==still perform poorly when evaluated on more complex, unseen problems== that require problem-solving skills beyond simply translating instructions into code. For example, competitive programming problems which require an understanding of algorithms and complex natural language remain extremely challenging. To address this gap, we introduce ==AlphaCode==, a system for code generation that can create novel solutions to these problems that require deeper reasoning. In simulated evaluations on recent programming competitions on the Codeforces platform, ==AlphaCode achieved on average a ranking of top 54.3% in competitions with more than 5,000 participants==. We found that three key components were critical to achieve good and reliable performance: (1) an extensive and clean competitive programming dataset for training and evaluation, (2) large and efficient-to-sample transformer-based architectures, and (3) ==large-scale model sampling to explore the search space, followed by filtering based on program behavior to a small set of submissions==.


