August 11, 2023
[[Meta AI Research]]
Paper: [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259)
#zotero 
Takeaway: A method/model that applies the **instruction backtranslation** method, using a fine-tuned language model and web corpus to construct instruction prompts for web documents (**self-augmentation**), then selecting high-quality examples from these generations (**self-curation**).

Relevance: ...

Aside: 
- The [[Self-Reward]]ing Language Models paper had critical things to say (see Obsidian references) about the LLM-as-a-Judge prompt used in this paper, where we had the model pick a score in a multiple-choice fashion, rather than using their explicit "add a point" criteria.
- I can sort of see how papers like this inspired Nous Research's [[Genstruct]] model, and the [[Evol-Amplify]] technique from Capybara.

----

Note:
- ...

Abstract
> We present a ==scalable method to build a high quality instruction following language model by automatically labelling human-written text with corresponding instructions==. Our approach, named ==instruction backtranslation==, starts with a language model finetuned on a small amount of seed data, and a given web corpus. ==The seed model is used to construct training examples by generating instruction prompts for web documents== (==self-augmentation==), and ==then selecting high quality examples== from among these candidates (==self-curation==). This data is ==then used to finetune a stronger model==. Finetuning LLaMa on two iterations of our approach yields a model that outperforms all other LLaMa-based models on the Alpaca leaderboard not relying on distillation data, demonstrating highly effective self-alignment.


# Paper Figures
- 
# Non-Paper Figures
- 