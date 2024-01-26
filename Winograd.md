This dataset was quoted in the GPT-3 paper, [[GPT-3 Paper {Language Models are Few-Shot Learners}|Language Models are Few-Shot Learners]]
A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences. It requires the use of world knowledge and reasoning for its revolutions.

Example:
> The city councilmen refused the demonstrators a permit because they `feared/advocated` violence.

If the word `feared` is used, then `they` refers to the city council. If `advocated` is used, it refers to the `demonstrators`.
This dataset assembles a set of such ==Winograd schemas== that are easily disambiguated by the human reader, not solvable by simple techniques, and ideally Google-proof.

The strengths of this challenge are that it's clear-cut, and that the answer to each schema is a binary choice.

It seems that models are able to achieve something like 85%+ accuracy, in ~2022.

