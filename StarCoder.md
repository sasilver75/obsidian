May 9, 2023 -- A huge assembly of researchers from 38 places, including [[HuggingFace]], [[Eleuther]], and more
Paper: [StarCoder: may the source be with you!](https://arxiv.org/abs/2305.06161)

A 15.5B code-generating model trained on 1T tokens from [[The Stack]] and fine-tuned on 35B Python tokens.
I believe [[The Stack]] was actually created for this explicit paper by [[Eleuther]].

Abstract
> The BigCode community, an open-scientific collaboration working on the responsible development of Large Language Models for Code (==Code LLMs==), introduces ==StarCoder== and StarCoderBase: ==15.5B parameter models with 8K context length==, infilling capabilities and fast large-batch inference enabled by multi-query attention. StarCoderBase is ==trained on 1 trillion tokens sourced from The Stack==, a large collection of permissively licensed GitHub repositories with inspection tools and an opt-out process. We fine-tuned StarCoderBase on 35B Python tokens, resulting in the creation of StarCoder. We perform the most comprehensive evaluation of Code LLMs to date and show that StarCoderBase outperforms every open Code LLM that supports multiple programming languages and matches or outperforms the OpenAI code-cushman-001 model. Furthermore, StarCoder outperforms every model that is fine-tuned on Python, can be prompted to achieve 40\% pass@1 on HumanEval, and still retains its performance on other programming languages. We take several important steps towards a safe open-access model release, including an improved PII redaction pipeline and a novel attribution tracing tool, and make the StarCoder models publicly available under a more commercially viable version of the Open Responsible AI Model license.

