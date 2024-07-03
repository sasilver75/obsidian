#article 
Link: https://substack.stephen.so/p/why-im-excited-about-dspy

Review: This article sucked, didn't really say anything at all besides "I don't like prompting, and it seems like DSPy gets good results."

----

The author has been working with LLMs and talking to people working with them.

Their first product was a solution to help people manage their prompt chains -- it was the result of our personal frustration managing prompts in complex pipelines, testing them, and trying to control different versions.

Two broad categories of products that achieved success in Generative AI:
1. Consumer/Prosumer "GPT Wrappers"
2. Infrastructure companies who support these use cases, and attempt to support more complex ones
	- (eg Hosting models with APIs, observability, testing, versioning, fine-tuning-as-service)

The *third*, much-less-successful (by traction) category is people trying to build towards more complex use cases.
- These people constantly told us about the difficulties of building complex pipelines

Interestingly, there's a large void between the number of companies in the successful and unsuccessful categories -- why aren't more "complex" use cases thriving.

The journey of building products with language models goes something like this:
1. It's easy to get started -- GPT4 gets you a proof of concept
2. You start trying to do more complex things, like retrieving your own data and augmenting the prompt with it before the generation (aka RAG)
3. You realize these things are a bit harder, so you turn to packages like `Langchain` which do a great job of making this really easy.
	- The tradeoff here with a "batteries included" framework is an abstraction layer that hides details, while still having hard-coded prompt strings.
4. You get a proof of concept (kind of) working
5. The PoC isn't working for *some* use cases, or has some issue that needs a change
6. You spend some time digging into the underlying abstraction, only to realize that it needs significant changes to fit this use case.
7. You rip apart the example, taking smaller primitives from popular packages and write your own!
8. You realize that with more moving components and an extensive amount of hardcoded prompts, you need version control, testing, and other services to support the inference calls to language models.
9. You finally get an example working and now you try to change your pipeline (or maybe OpenAI decides to re-lobotomise the model, or you switch models for cost reasons, etc) and ...
10. Things go kaboom! 😭💥

The problem starts when developers move from creating simple products, often with a single fixed prompt, to attempting to construct complicated pipelines involving several requests, yet still rely on fixed prompts.

If this describes you, don't worry!

Let's have a hypothetical pipeline:
![[Pasted image 20240401205926.png]]
Above:
- With each step in our pipeline having an accuracy of .9, that means that the overall pipeline accuracy is only going to be (.9^6) = 53%! That's not workable!
	- ((I realize that some things don't really have an "accuracy" score))

Let's say that we switch up our Language Model to [[Mistral]] or something, and improve most of our accuracy scores:

![[Pasted image 20240401210305.png]]
Okay, nice... that's a little better...
But there's still a laggard there.

What would happen if we then tweaked our prompts?
![[Pasted image 20240401210424.png]]

((Honestly I don't see the brittleness from these examples))

We iterate until we get something right, and then when wee want to change something -- Kaboom!
It leads to a fragile system that's increasingly difficult to work with as it grows in complexity

----

[[DSPy]] is a ==framework for algorithmically optimizing LM prompts and weights==, which shines when language models are used one or more times in a pipeline.
- The ==prompts== and ==weights== here is important
- DSPy can be used to optimize the prompts *without* finetuning the actual model (weights)

==Instead of focusing on hand crafted prompts that target specific applications, DSPy has general purpose *modules* that learn to prompt (or finetune) a language model==

This solves the fragility problem by allowing you to simply ==recompile== the pipeline when you make changes!

The results so far seem to be worth paying attention to:

![[Pasted image 20240401211049.png|450]]

I think the future of programming with language models looks a lot more like DSPy -- with less emphasis on prompting, and more on programming.

These increasingly complex use cases can only be solved if we spend more time programming, and not fixing an increasingly fragile pipeline.
- It sort of feels like *prompting* and *programming* are two distinct tasks that don't really *gel*.
	- This approach is an order of magnitude improvement for developer experience.





