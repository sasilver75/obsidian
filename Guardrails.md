Guardrails validate the output of LLMs, ensuring that the output doesn't just *sound good*, but is also syntactically correct, factual, and free from harmful content. It also includes guarding against adversarial input.
- Shape/Syntax: We may want to require output to be in a specific JSON schema so that it's machine-readable. We may need code that's generated to actually be executable.
- Alignment: We may want to ensure that output isn't harmful, verify it for factual accuracy, or ensure coherence with the context provided.

See:
- The [Guardrails](https://github.com/guardrails-ai/guardrails) package, or Nvidia's NeMo-Guardrails (see [difference](https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=2023+Year+in+Review%20-%2012699108))
- Models like [[LLaMA Guard]]

Useful and practical strategies:
- ==Structural guidance==: Apply guidance wherever possible. It provides direct control over outputs and offers a more precise method to ensure that output conforms to a specific structure or format.
- ==Syntactic guardrails==: Include checking if categorical output is within acceptable choices, or if numeric input is in expected range.  If we generate SQL, we can ensure it's free from syntactic errors.
- ==Content safety guardrails==: Ensure the output has no harmful or inappropriate content. It can be as simple as checking against a list of naughty words, or using separate profanity detection models.
- ==Semantic/factuality guardrails==: These confirm that the output is semantically relevant to the input. If we're generating a two-sentence summary of a movie based on its synopsis, we can validate that the produced summary is semantically similar to the output, or have another LLM ascertain if the summary accurately represents the provided synopsis.
- ==Input guardrails==: These limit the types of *input* the model responds to, helping it to avoid responding to inappropriate or adversarial prompts which would lead to generating harmful content.