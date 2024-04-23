May 9, 2023 -- A huge assembly of researchers from 38 places, including [[HuggingFace]], [[Eleuther]], and more
Paper: [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)
HuggingFace Dataset: [The Stack](https://huggingface.co/datasets/bigcode/the-stack)

Dataset of 6.4TB (320x larger than Wikpiedia), ==200B tokens==, that has permissive licenses. Cleaned of license files, image blobs, etc.  This is a code dataset, compared to [[The Pile]], which is a text dataset.

Released as part of the widely-collaborative BigCode project, which resulted in the [[StarCoder]] code model. I believe [[The Stack]] was primarily [[Eleuther]]'s contribution.

In late Feb 2024, [[The Stack v2]] was released, which is about 5x larger at ~900B tokens (~67.5TB).

Abstract from the [[StarCoder]] paper
> The BigCode community, an open-scientific collaboration working on the responsible development of Large Language Models for Code (==Code LLMs==), introduces ==StarCoder== and StarCoderBase: ==15.5B parameter models with 8K context length==, infilling capabilities and fast large-batch inference enabled by multi-query attention. StarCoderBase is ==trained on 1 trillion tokens sourced from The Stack==, a large collection of permissively licensed GitHub repositories with inspection tools and an opt-out process. We fine-tuned StarCoderBase on 35B Python tokens, resulting in the creation of StarCoder. We perform the most comprehensive evaluation of Code LLMs to date and show that StarCoderBase outperforms every open Code LLM that supports multiple programming languages and matches or outperforms the OpenAI code-cushman-001 model. Furthermore, StarCoder outperforms every model that is fine-tuned on Python, can be prompted to achieve 40\% pass@1 on HumanEval, and still retains its performance on other programming languages. We take several important steps towards a safe open-access model release, including an improved PII redaction pipeline and a novel attribution tracing tool, and make the StarCoder models publicly available under a more commercially viable version of the Open Responsible AI Model license.

