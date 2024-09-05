https://www.yitay.net/blog/model-architecture-blogpost-encoders-prefixlm-denoising
Authored by [[Yi Tay]], chief scientist at [[Reka AI]]

It's not like this author can't speak english, but something about the way that he constructs his essay makes it difficult to follow. He doesn't offer explanations where I think there ought to be one.

----

People who worked in NLP 5 years ago are left thinking about where encoder models like [[BERT]] or encoder-decoder models like [[T5]] went? If they worked, why didn't people just make them bigger?

> "If xBERT worked well, why didn't people make it bigger?... Is the theory that denoising gets too easy for big models?" - [[Sasha Rush]]

There are mainly three overarching paradigms of model architectures in the past few years:
- [[Encoder-Only Architecture]] ([[BERT]])
- [[Encoder-Decoder Architecture]] ([[T5]])
- [[Decoder-Only Architecture]] ([[GPT]])

Understand: Encoder-Decoder architectures are still autoregressive models! Its decoder is still literally and fundamentally still a causal decoder, which then cross-attends to the encoder.

A variant of this is a ==Prefix Language Model== or ==PrefixLM== architecture, which does almost the same thing *minus the cross attention* (and some other small details like sharing weights between encoder/decoder plus not having no encoder bottleneck).
- PrefixLMs are sometimes known as *non-causal* decoders.

People assume that encoder-decoder models have to be denoising models partly because of the *overly representative* T5 model, but this isn't always true -- you can train them with a regular LM modeling task.

Generally speaking, an Encoder-Decoder of 2N parameters has the same compute cost as a decoder-only model of N parameters.
- ((I think this is because the decoder generates the output token by token, whereas the encoder-decoder encodes the entire input in a single forward pass and then similarly generates the output token by token. So it's just like N+1 passes.))

Denoising objective is any variation of the "span corruption" task, which is sometimes known as "infilling" or "fill in the blank" -- there are some variations on how to express it (regarding span length, randomness, sentinel tokens, etc.).

The denoising objective in BERT-style models is mostly "in place," but the modern way is to do it "==T5-Style==," where masked tokens are "moved to the back" via data transformation, for the model to predict them. This can be processed by an encoder-decoder *or* a decoder-only model.

The main goal of pretraining is to build a useful internal representation that can be aligned for downstream tasks... the better the internal representation, the easier to use these learned representations for anything useful later.
- Simple next word prediction objectives are known to learn useful internal representations very well, and has been the bread and butter of the LLM revolution.
	- The question is whether denoising objectives are just as good?


We know that T5-11B works pretty well even after being aligned/SFT'd. (Flan-T5 XXL's MMLU score was 55+, which was good for that scale and time).
- So we can make the conclusion that the pretraining -> alignment process works relatively well even for the denoising objective.


My take is that the denoising objectives are great but pretty insufficient as standalone objectives.
- In denoising objectives, only a small amount of tokens are being masked and get learned as a result (ie taken into account in the loss), whereas in regular causal language modeling, this is close to 100%... making for a pretty low sample efficiency per FLOP which ==makes denoising objectives hugely disadvantaged on a flop-basis comparison.== ((?))
- Another drawback is that denoising objectives are more unnatural than regular language modeling, since it reformats the input/output in a strange way, ==making them a little awkward for few-shot learning.==

==Hence, I believe denoising objectives should pretty much only be used as a complementary objective to regular language modeling.==

----

The gradual phasing out of BERT-like models was an interesting phase that not many people talk about these days... it was largely a matter of unification and shift in task/modeling paradigms.

==The real deprecation of BERT models was because people wanted to do all tasks at once, which led to a better way of doing denoising, using autoregressive models.==

In 2018-2021 there was a paradigm shift of single-task finetuning to *massively multi-task models*. 
- "It was simply so hard to do this with BERT."
- To be even more concrete, ==encoder-decoder and decoder-only models were able to express multiple tasks at once without the need for task-specific classification heads==.

Authors also began to find that yanking out the encoder from an encoder-decoder performed just as competitive as a BERT encoder, and retains the same bidirectional attention benefit.

----

Pretraining task mixtures (denoising, language modeling) can be stacked sequentially and doesn't necessarily have to be mixed concurrently.
- [[FLAN-T5]] originally trains on 1T span corruption tokens and switches out to 100B tokens of prefix language modeling objective before flan instruction tuning.

Anecdotal experience is that denoising objectives learn representations that are better at *certain classes of tasks,* sometimes in a more sample-efficient way.
- U-PaLM paper showed how a small amount of span corruption up-training changes behavior and emergence on a set of BIG-Bench tasks.
- Finetuning models trained with this (SC) objective generally result in better SFT models, especially at smaller scales.

==When it comes to single-task finetuning, you can see the OG [PaLM-1](https://arxiv.org/abs/2204.02311) 62B model gets defeated by a much smaller T5 model. Bidirectional attention + denoising objective packs a punch at a relatively small scale!==

---

Encoder-Decoder Architectures (ProCon vs regular Decoder-Only models)
+: Encoder side is not restricted by a causal mask.
-: Inputs and targets have to have fixed allocated budgets.e.. if the input is 1024 tokens, the encoder side has to be padded to this value, which causes a lot of potential for wasted compute.
- In PrefixLM, input and targets can be directly concatedd which mitigates this probelem.

---

==Key takeaways:==
1. EncDec and Dec models are both autoregressive models with only implementation-level differences and pros/cons. They are subtly different inductive biases. Optimal usage depends on downstream use-cases and pretty much application constraints.
	- For most LLM usage and niche use-cases aside, BERT-style encoder models are mostly considered deprecated.
2. ==Denoising objects are mostly *complementary* to causal language modeling==! They're sometimes used as "supporting objectives" in pretraining -- this happens frequently in code models )eg code infilling), but it's not uncommon for general purpose models to pretrain with *some* denoising objective.
3. Bidirectional attention helps a lot at saller scales, but is generally optional at larger model scales. This is mostly anecdotal.

BERT models were deprecated in favor of more flexible forms of denoising (autoregressive) T5 models... largely due to a paradigm unification where people would like to perform any task with a general-purpose model, as opposed to a task-specific one.

















