Link: https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-iii-strategy/

This is part 3 (on Strategy) of a series with two previous parts

On Tactics [[What we learned from a Year of Building with LLMs (Part 1) (May 28, 2024) {Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, Shreya Shankar}]]

On Operations [[What we Learned from a Year of Building with LLMs (Part 2) {Eugene Yan, Bryan Bischof, Charles Frye,  Hamel Husain, Jason Liu, Shreya Shankar}]]

---

We've talked about granular Tactics (Part 1) and the higher level processes of Operations (Part 2)
But where do the objectives that we're trying to achieve come from? ==Strategy== answers the "What" and "Why" questions behind the "how" of tactics and operations.

We have some opinionated takes, and suggest a roadmap.
We aim to answer the following questions:

1. ==Building vs Buying==: When should you train your own models, and when should you use APIs? It depends!
2. ==Iterating to Something Great==: How can you create a lasting competitive edge that goes beyond just using the latest models? We need to deliver memorable, sticky experiences.
3. ==Human-Centered AI==: How can you effectively integrate LLMs into human workflows to maximize productivity and happiness?
4. ==Getting Started==: What are essential steps for teams embarking on building an LLM product? Start with prompt engineering, evaluation, and data collection.
5. ==The Future of Low-Cost Cognition==: How will the rapidly decreasing costs and increasing capabilities of LLMs shape the future of AI applications?
6. ==From Demos to Products==: What does it take to go from a compelling demo to a reliable, scalable product?

---

# (3/3) Strategy: Building with LLMs without getting out-maneuvered

Successful products require thoughtful planning and tough prioritization, not endless prototyping or following the latest model releases or trends. Let's examine tradeoffs and suggest a playbook:

## No GPUs before PMF
- Your product needs to be more than just a thin wrapper around someone else's API - but mistakes in the opposite direction can be even more costly!
- ==Training from scratch (almost) never makes sense==
	- As much as it seems like everyone else is doing it, developing and maintaining ML infrastructure takes a lot of resources including gathering data, training and evaluating models, and deploying them. It's very likely that you don't have the need for this, *especially* if you're pre PMF.
	- Maintaining competitive utility for your model requires continued investment and staffing.
- ==Don't fine-tune until you've proven it's necessary.==
	- For most organizations, fine-tuning is driven more by excited FOMO than by clear strategic thinking. LLM-powered applications aren't a science fair project.
	- ==Organizations invest in fine-tuning too early, trying to beat the "just another wrapper" allegations==. In reality, fine-tuning is heavy machinery, to be deployed only after you've collected plenty of examples that convince you that other approaches won't suffice.
- Managed services aren't right for every use case though
	- Self-hosting may be the only way to use models without sending confidential/private data out of your network (healthcare, finance)
	- Circumvents limitations imposed by inference providers like rate limits, model deprecations, and usage restrictions.

## Iterate to something great
- ==The model isn't the product, the system around it is.==
- For teams that aren't building models, the rapid pace of innovation is a boon as they migrate from one SOTA model to the next, chasing gains in context size, reasoning capability, and price-to-value to build better and better products.
- ==Focus your effort on what's going to provide lasting value:==
	- ==Evaluation chassis==: To reliably measure performance on your task across models.
	- ==Guardrails==: To prevent undesired outputs no matter the model.
	- ==Caching==: To reduce latency and cost by avoiding the model altogether.
	- ==Data flywheel==: To power the iterative improvement of everything above.
- Think about where it's a good use to spend your time: Lots of teams invested in building custom tooling to validate structured output from proprietary models -- deep investment here is not a good use of time (Providers now provide this, libraries exist).
- Building product that tries to be everything to everyone is a recipe for mediocrity; create compelling products by building memorable, sticky experiences.
- Consider a generic RAG system that aims to answer any question a user might ask. The lack of specialization means that the system can't prioritize recent information, parse domain-specific format, or understand the nuances of specific tasks.
- Build LLMOps for the right reasons: faster iteration
	- Production monitoring and continual improvement, linked by evaluation.
- Tools like [[LangSmith]], Log10, LangFuse, [[Weave]] from W&B, HoneHive, and more promise to not only collect and collate data about system outcomes in production, but also to leverage them to improve those systems by integrating deeply with development.

#### Don't build LLM features you can buy!
- ==Most successful businesses aren't LLM businesses. Simultaneously, most businesses have opportunities to be improved by LLMS.==
- Consider a few misguided ventures that waste your team's time:
	- Building a custom text-to-SQL capabilities for your business
	- Building a chatbot to talk to your documentation
	- Integrating your company's KB with your customer support chatbot.
- ==Investing valuable R&D resources on general problems being tackled en masse by the current Y Combinator batch is a waste.==

#### AI in the loop; humans at the center
- Right now, LLM-powered applications are brittle. They require an incredible amount of safe-guarding, defensive engineering, and remain hard to predict.
	- But when tightly scoped, these applications can be very useful and accelerate user workflows.
- We should center humans and ask how LLMs can support their workflow.

## Start with prompting, evaluations, and data collection
- Previous sections have delivered firehoses of techniques and advice, but if you're just getting started, where should you begin?

#### Prompt Engineering Comes First
- Use all techniques we discussed in the tactics section before ([[Chain of Thought|CoT]], [[Few-Shot Prompting]], Structured Inputs/Outputs). Prototypes with the most highly capable models before trying to squeeze performance out of weaker models.
- Only if prompt engineering cannot achieve the desired level of performance should you consider fine-tuning.
#### Build Evals and kickstart a Data Flywheel
- Even teams that are getting started need evals; effective evals that are ==specific to your task and mirror the intended usecases== are important!
- Start with ==unit testing== -- simple assertions detect known or hypothesized failure modes and help drive early design decisions.
- Consider other [task-specific evaluations](https://eugeneyan.com/writing/evals/) for classification, summarization, etc.
- Unit tests and model-based evaluations are useful, but don't replace the need for human evaluation. Have people use your model/product and provide feedback.
	1. Human evaluation to assess model performance and/or find defects
	2. Use the annotated data to finetune the model or update the prompt
	3. Repeat

## The high-level trend of low-cost cognition
- In the four years since OpenAI's davinci model was launched as an API, the cost for running a model with equivalent performance on that task at a scale of 1M tokens has dropped from $20 to less than 10 cents, with a halving time of just six months.
- The trends are new, but there's little reason to expect this process to slow down in the next few years. There are many fronts on which innovations will continue to be found, whether it's data, algorithms, or compute.

## Enough 0 to 1 Demos, It's time for 1 to N products
- Building LLM demos are a ton of fun -- with a few lines of code, a vector DB, and a carefully crafted prompt, we create ✨magic✨!
	- But there's a world of difference between a demo and a product at scale.
- The first self-driving car with a neural net was in 1988, and 25 years later in 2013, Andrej Karpathy took his first demo ride in a Waymo. 11 years after that and I'm hailing it on an app in Los Angeles.














