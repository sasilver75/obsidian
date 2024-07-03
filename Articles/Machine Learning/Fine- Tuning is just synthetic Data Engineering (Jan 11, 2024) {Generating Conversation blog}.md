#article 
Link: https://generatingconversation.substack.com/p/fine-tuning-is-just-synthetic-data

----

Over the course of 2023, we saw the ==proliferation of fine-tuning APIs==, starting with OpenAI's, but quickly followed by third-party service providers like TogetherAI, Anyscale, Lamini, etc.

==Now, fine-tuning models is turnkey: Upload a dataset, pick a model, and press "Go"==
- A few hours later, you have an inference endpoint.

There's certainly a ton of complex engineering going on under the hood, but as a user, you're exposed to none of it!

...But where do you get the data?
As with many things in AI, the ==hardest questions come back to data engineering!==

# Synthetic Data in King
- Synthetic data has turned out to be a powerful tool!
- =="Read this document, and write 10 question-answer pairs relevant to it."==
- You take these pairs and train *smaller* models on it, which won't have the *general purpose reasoning capabilities* that the larger model has, but will be very good at following the pattern of examples you provided for it.
- ==So we use a larger model for dataset generation in order to finetune a model, and then as a result get cheaper and faster inference moving forwards.==
	- (You can still use RAG techniques to maintain freshness along as the basic facts and structure you fine-tuned the model on remain constant)

# But again, (synthetic) data generation
- Unfortunately, we've just done a whole lot of handwaving.
- ==We still need to make sure that our data is high quality==, and that's a big stumbling block! GPT-4 and similar models help us generate data, but even these models are garbage in, garbage out.
- This is where data engineering comes in:
	- How long should the individual input datas be for dataset generation?
	- How many examples is enough?
	- Is there such a thing as too-small a dataset for fine-tuning?
	- How do you ensure sufficient *diversity* in your examples? Inundating the model with the same few examples repeatedly is terrible for model performance -- you should vary the type and order of examples significantly!


# It's simple until it isn't
- These fine-tuning services we mentioned before are powerful and cost-effective, but they expose relatively few knobs for users to manage!
- The author imagines that most of these services are using [[Low-Rank Adaptation|LoRA]] to fine-tune models nad rapidly swap them at inference time!
- But you may eventually want to move to your own models, for any of the following reasons:
	- Cost
	- Quality
	- Privacy reasons

When you get to that point, consider:
1. Where do I get GPUs, in this shortage?
2. Do I use LoRA (or other PEFTs) or do full weight updates? (LoRA is more efficient but leads to slightly worse model quality)
3. What fine-tuning implementation strategy should I use?
4. What about deployments? Do CSPs give me a bring-my-own-model deployment mechanism?