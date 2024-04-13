A ==dataset== (and model) from [[Nous Research]] with roughly ==15,000 multi-turn instruction-following examples==.

This was the culmination of experimenting with the most useful insights derived from synthesis techniques like:
1. [[Evol-Instruct]] (used for [[WizardLM]])
2. [[Orca]]
3. [[Alpaca]]
4. [[Vicuna]]
5. Lamini
6. FLASK

Along with intuitions from over 3 years of doing data curation for dozens of models in text and audio modalities across different architectures.

The result is a multi-turn synthetic conversation generation method (that might still be) called ==Amplify-Instruct== -- the first resulting dataset using this method is called Capybara!
- A focus on information diversity across a wide range of domains, and multi-turn conversations that strongly emphasize reasoning, logic, and extrapolation about a wide variety of subjects -- and many great examples of conversations delving into obscure sub-topics and rabbit holes across pop-culture and STEM, while still maintaining natural prose!

A first test of Capybara shows that with less than 20,000 of these high-quality examples, we can reach beyond the HF leaderboard scores of many popular models that were trained (on the same base model) with >5x the amount of data!
- ((==Given the same base model, models resulting from finetuning on this dataset are as good as models finetuned on 5x+ more data!==))