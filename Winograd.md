
June 10, 2012
Paper: [The Winograd Schema Chalenge](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf)


A ==Winograd Schema== is a pair of sentences that differ in only one or two words and that contain an ambiguity that is resolved in opposite ways in the two sentences. It requires the use of world knowledge and reasoning for its revolutions. It's basically coreference resolution when there are ambiguities in a sentence that require *world knowledge* to resolve.

Example:
> The city councilmen refused the demonstrators a permit because they `feared/advocated` violence.

If the word `feared` is used, then `they` refers to the city council. If `advocated` is used, it refers to the `demonstrators`.

This dataset assembles a set of ==273== of such ==Winograd schemas== that are easily disambiguated by the human reader, not solvable by simple techniques, and ideally Google-proof..
The strengths of this challenge are that it's clear-cut, and that the answer to each schema is a binary choice.

It was succeeded by the [[Winogrande]] dataset and benchmark

Abstract
> In this paper, we ==present an alternative to the Turing Test== that has some conceptual and practical advantages. ==A Winograd schema is a pair of sentences that differ only in one or two words and that contain a referential ambiguity that is resolved in opposite directions in the two sentences==. We have compiled a collection of Winograd schemas, designed so that the correct answer is obvious to the human reader, but cannot easily be found using selectional restrictions or statistical techniques over text corpora. A contestant in the Winograd Schema Challenge is presented with a collection of one sentence from each pair, and required to achieve human-level accuracy in choosing the correct disambiguation.
> 

![[Pasted image 20240411230118.png]]