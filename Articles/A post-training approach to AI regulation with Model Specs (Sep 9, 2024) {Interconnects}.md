https://www.interconnects.ai/p/a-post-training-approach-to-ai-regulation

---

A central point of AI regulation (eg [[SB-1047]]) has been model size, or when an AI system uses more than a certain amount of compute or money to be trained.

A common chorus is that we should regulate the applications of AI systems, rather than the models themselves... with the argument being that the ==most== harm comes from misuse, and amplifications of existing harms, rather than ==new harms being unlocked== by latent capabilities of the model.
- ((I don't know how I feel about this. If you regulate the models themselves, then you can take certain capabilities and possibilities of misuse off the table. The goal isn't to prosecute people, it's to prevent harm. If you just regulate the use of models, and if misuse risks are real, then you're putting (what can be used as) a loaded pistol on the table and saying "If anyone uses this for great harm, we'll sue you!" If your goal is to prevent harm, that's not really an honestly tenable position, is it? I guess the question is should laws prevent the possibility of harm, or should they just punish harm when it happens?))
- But I guess the claim here is that most of the harms that we can worry about are already possible using the latent ability of current models, so focusing laws on the misuse of current models is better? But certainly there are new harm categories (eg CBRN) that aren't really viably enabled with current models, and so we should regulate at the model/lab level for those, right?


The ==Model Spec==, currently only released by OpenAI, is ==a document that conveys the goal behavior of a model served to customers==. It was written by members of the post-training team and product teams at [[OpenAI]], such as co-founder [[John Schulman]] (now at Anthropic). The document ==contains a list of principles and examples for desired behavior, including things that could be the source of disagreement and behaviors that may not reliably work in current models==.

> Sama: We will listen, debate, and adapt this over time, but i think it will be very useful to be clear when something is a bug vs. a decision.

> Joanna Jang, OAI Product: Principles are easier to debate and get feedback on, vs. hyper-specific screenshots or abstract feel-good statements.

Model Specs can bridge audiences. Being clear about the intention of the models is useful for addressing both short- and long-term types of risk.

[[Constitutional AI]] is a method for getting many different behaviors out of a model and it is _not clear_ if the downstream behaviors of a model are calibrated to the principles used in the training data.
- Releasing Constitutional AI principles along with ==_why_== they were used begins to be useful for oversight.


