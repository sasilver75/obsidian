#article  #premium 
Link: https://www.noemamag.com/what-ai-can-tell-us-about-intelligence/

---------
If there's one constant in the field of AI, it's that there's always both breathless hype and scornful naysaying.

The dominant technique in contemporary AI is Deep Learning -- massive self-learning algorithms that excel at discerning and utilizing patterns in data.

Since their inception, critics have prematurely argued that NNs have run into an ==insurmountable wall== -- and every time, it has proved a temporary hurdle which were the lack of training data and computing power (and also simplified programs).

The introduction of specialized compute hardware (starting with GPUs) and self-supervised learning, which increased the amount of data available, has resulted in incredibly impressive systems.

Today's seemingly insurmountable wall is ==symbolic reasoning== -- the capacity to manipulate symbols in the ways familiar from algebra or logic. As we learned as children, solving math problems involves a step-by-step manipulation of symbols, according to strict rules.

Gary Marcus disagrees, but many DL researchers are convinced that DL is already engaging in symbolic reasoning, and will continue to improve at it.

The heart of this debate are ==two different visions of the role of symbols in intelligence, both biological and mechanical: One holds that symbolic reasoning must be hardcoded from the outset, and the other holds that use of symbols can be *learned* through experience==

The stakes are thus not just about the most practical way forward, but also how we should understand human intelligence, and thus how we should pursue human-level artificial intelligence.

# Kinds of AI
- The field of AI got its start by studying this kind of reasoning, typically called Symbolic AI, or "Good Old-Fashioned" AII (GOFAI).
- Distilling human expertise into a set of rules and facts turns out to be very difficult, time-consuming, and expensive. This was called the =="knowledge acquisition bottleneck"== -- ==it proved impossible for lowly humans to write rules governing every pattern, or define symbols for vague concepts==.

==This is precisely where neural networks excel: discovering patterns and embracing ambiguity.==
- Neural networks are a collection of relatively simple equations that learn a function designed to provide the appropriate output for whatever is inputted to the system -- which can be things like whether a new object is a chair, simply by how close it is to a cluster of other chair images.

These networks can be trained precisely because the function implemented are differentiable -- put differently, ==if Symbolic AI is akin to the discrete tokens used in symbolic logic, neural networks are the continuous functions of calculus.==

This fluidity poses problems, however, when it comes to strict rules and discrete symbols -- when we're solving an equation, we want the *exact* answer, not an approximation!

This, in contrast, is where Symbolic AI shines!

Gary Marcus recommends simply combining the two, inserting a hard-coded symbolic manipulation module on top of a pattern-completion DL module, in a "hybrid" system.

But the debate turns into whether symbolic manipulation needs to be *built into* the system, where the symbols and capacity for manipulating are designed by humans and installed as a module (see: knowledge bottleneck).

# Symbolic Reasoning in Neural Nets
- The neural network approach has traditionally held that we don't need to hand-craft symbolic reasoning, but can instead learn it.
	- Training a machine on examples of symbols engaging in the right kinds of reasoning will allow it to be learned as a matter of abstract pattern completion.
- Contemporary language models (eg GPT, LaMBDA) show the potential of this approach -- they're capable of impressive abilities to manipulate symbols, displaying some level of common-sense reasoning, compositionally, multilingual competency, etc... ==but they do not do so reliably!== (Eg if you try to ask it to draw a beagle in a pink harness chasing a squirrel, sometimes you get a pink beagle, or a squirrel wearing a backpack.) Will this be "fixed" with scale? Remains to be seen.

Gary Marcus, in contrast, assumes that symbolic reasoning is *all or nothing* -- DALL-E isn't *actually* reasoning with symbols, thus the numerous failures in LLMs show they aren't *genuinely* reasoning, but are simply going through a pale imitation.

==From one perspective, the problems of DL are hurdles, and from another perspective, walls.==

Marcus's critique of DL stems from a related fight in cognitive science concerning how intelligence works, and, with it, what makes humans unique.

His ideas are in line with a prominent "==nativist==" school in psychology, which holds that many features of cognition are innate -- effectively, that we're largely born with an intuitive model of how the world works.
- I think this is a [[Noam Chomsky]] thing

A central feature of this innate architecture is a capacity for symbol manipulation -- for Marcus, this symbol manipulation capacity grounds many of the essential features of common sense: rule-following, abstractions, causal reasoning, re-identifying particulars, generalization, and a host of other abilities.

==There's an alternate==, ==empiricist view== that inverts this: Symbolic manipulation is a rarity in nature, primarily arising as a learned capacity for communicating acquired gradually by our hominin ancestors over the last two million years. In this view, the primary cognitive capacities ... says that the vast majority of complex cognitive abilities are acquired through a general, self-supervised learning capacities. It also assumes that most of our complex cognitive capabilities don't run on symbolic manipulation, but instead of *simulating* various scenarios and predicting the best outcomes. This empiricist view treats symbols/symbolic manipulation as simply another learned capacity, one acquired by the species as humans increasingly relied on cooperative behavior for success.

If the DL advocates and empiricists are right, it's instead the idea of inserting a module for *symbolic manipulation* (a la Gary Marcus) that's confused!
- In this case, DL systems are already engaged in symbolic reasoning, and will continue to improve at is as they become better at satisfying constraints through more multimodal self-supervised learning.

As big as the stakes are, though, it's important to note that many issues raised in these debates are peripheral.
- There are claims that high-dimensional vectors in DL systems should be treated like discrete symbols (probably not), whether the lines of code needed ot implement a DL system make it a "hybrid" system, whether winning at complex games require handcrafted, domain-specific knowledge or whether we can learn it







