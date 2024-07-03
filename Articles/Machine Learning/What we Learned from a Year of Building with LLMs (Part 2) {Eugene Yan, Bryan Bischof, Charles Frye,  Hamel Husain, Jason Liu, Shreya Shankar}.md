https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/

Part 1: [[What we learned from a Year of Building with LLMs (Part 1) (May 28, 2024) {Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, Shreya Shankar}]]
- Part 1 talked about "Tactics," we'll talk about the *operational* aspects of building LLMs that sit between tactics (Part 1) and strategy (Part 3)

---

Actually operating an LLM application raises questions that are familiar from operating traditional software systems, often with a novel spin to keep things spicy.

==The questions that we'll cover are across four parts:==
1. ==Data==: How often should you review LLM I/Os? How to measure test/prod skew?
2. ==Models==: How do you integrate LMs into the rest of your stack? How do you version models and migrate between models and versions?
3. ==Product==: When should design be involved in the development process, and why is the answer "as early as possible?" How do you design user experiences with rich human-in-the-loop feedback? How do you calibrate product risk?
4. ==People==: Who should you hire to build a successful LLM application, and when can you hire them? How do you build a culture of experimentation?

# (2/3) Operations: Developing and Managing LLM Applications and the Teams that build them

## (1/4) Data
- Output data is the only way to tell whether the product is working or not. ==ALL of the authors of this article== focus tightly on the data, ==looking at both inputs and outputs for several hours a week== to better understand the data distribution's modes, edge cases, and the limitations of the model.
#### Check for Development/Prod Skew
- A common source of errors in traditional ML is ==train-serve skew==, which happens when the data used for training differs from what the model encounters during production, causing accuracy to suffer.
- There are two types of LLM development-prod skew:
	- ==Structural==: Includes issues like formatting discrepancies, like differences between a JSON dictionary with list-type values vs a JSON list, inconsistent casing, errors like typos or sentence fragments.
	- ==Content-based/Semantic==: Differences in the meaning or context of data.
- It's good to periodically measure skew:
	- ==Simple metrics== like the length of inputs and outputs, or specific formatting requirements are straightforward ways to track changes.
	- For more "advanced" drift detection, ==consider clustering embeddings of I/O pairs to detect semantic drift==, like shifts in the topics that users are discussing, which could indicate that they're exploring areas the model hasn't been exposed to before.
- When testing changes, like new prompt engineering, ensure that holdout/evaluation datasets are current and reflect the most recent types of user interactions!
	- If typos are common in production inputs, they should also be present in the holdout data.

Generally, you should be regularly reviewing your model's outputs; a practice colloquially known as "==vibe checks==."
- You should be looking at samples of LLM I/Os every day!
- An iterative process of evaluation, re-evaluation, criteria updating, and prompt adjustment is likely necessary.

==You might want to consider running pipelines multiple times for each input in your testing dataset==, and analyze *all* outputs; this increases the likelihood of catching anomalies that might occur only occasionally.

You should be ==logging LLM inputs and outputs!==
- By ==examining a sample of these logs daily==, we can quickly identify and adapt to new patterns or failure modes.
- ==When we spot a new issue, we can immediately write an assertion or eval around it, and update our evaluation criteria if needed.==s
- Ideally this attitude is ==socialized==, for example by adding review or annotation of I/Os to your on-call rotation!

## (2/4) Models

When newer, better models drop, we should be prepared to update our products as we deprecate old models and migrate to newer models.
- Generate structured output to ease downstream integrations. Structured outputs are good, and tools like [[Instructor]] and Outlines can help you coax structured output from LMs!
	- Postel's law: "Be liberal in what you accept (arbitrary natural language) and conservative in what you send (typed, machine-readable objects)"
- ==Migrating prompts across models is a pain in the ass==
	- Our carefully crafted prompt might work superbly with one model, but fall flat with another. This can happen when we're switching between various model providers, as well as when we upgrade across versions of the same model.
	- Having reliable, automated evaluations helps with measuring task performance before and after painful migrations, and reduces the (still needed) effort for manual verification.
	- ==Remember to version and pin your models!== This can help avoid unexpected changes in model behavior, which could lead to customer complaints about issues that may crop up when a model is swapped, such as overly-verbose outputs or other unforeseen failure modes.
- Choose the smallest model that gets the job done.
	- The benefits of smaller models are lower latency and cost; techniques like CoT, n-shot prompting, and In-Context learning can help models punch above their weight and solve your problems.
	- If needed, finetuning of smaller models is also a good solution to increase performance.
	- Even lightweight models like [[DistilBERT]] (67M parameters) are a surprisingly strong baseline, or the 400M parameter [[DistilBART]], which, when finetuned, was able to surpass most LLMs at less than 5% of the latency and cost!

## (3/4) Product
- While new technology offers new possibilities, the principles of build great products are timeless. We don't have to reinvent the wheel on product design.
- You should involve design early and often to help you think deeply about how your product can be built and presented to users.
- What job is the user asking this product to do for them? Is that job something a ChatBot would be good at? How about autocomplete? Maybe something completely differently?
- Design your UX for human-in-the-loop; ==Allow users to provide feedback and corrections easily, so we can improve the immediate output and collect valuable data to improve our models!==
	- The pattern of *suggestion, user validation, and data collection* is commonly seen in several applications:
		- Coding assistants: Users can accept a suggestion (strong positive), accept and tweak a suggestion (positive), or ignore a suggestion (negative).
		- Midjourney: Users can choose to upscale and download the image (strong positive), vary an image (positive), or generate a new set of images (negative)
		- Chatbots: Users can provide thumbs up (positives) or thumbs down (negative) on response, or choose to regenerate a response if it was really bad (strong negative).
- ==Calculate your risk tolerance based on the use case==! For customer-facing chatbots offering medical or financial advice, we need a very high bar for safety and accuracy if we don't want to cause harm and erode trust!

## (4/4) People
- Focus on process, not tools.
- There's a growing industry of LLm evaluation tools that offer "LLM Evaluation in a Box" with generic evaluators for toxicity, conciseness, tone, etc. Many teams adopt these without thinking critically about tthe specific failure modes of their domains.
- Authors speak very favorably about [[EvalGen]], which focuses on teaching users the process of creating domain-specific evals by involving the user in every step of the way:
- ![[Pasted image 20240606165214.png|300]]

AI engineers should seek to understand the processes before adopting tools.

Always be experimenting
- ML products are deeply intertwined with experimentation; not just A/B RCT kind, but frequent attempts at modifying the smallest possible components of your system and doing offline evaluation.
	- ==The better your evals, the faster you can iterate on experiments, and thus the faster you can converge on the best version of your system!==

It's common to try different approaches to solving the same problem, because experimentation experimentation is so cheap now, between prompt engineering and synthetic data generation.

During product/project planning, ==set aside time for building evals and running multiple experiments!==
- During roadmapping, don't underestimate the time required for experimentation; expect to do multiple iterations of development and evaluation before getting the greenlight for production.

==Don't fall into the trap of "AI engineering is all I need"==
- It's a new title, and there's an initial tendency to overstate the capabilities associated with these roles - this often results in painful correction as the actual scope of these jobs become clear.
- Building ML/AI products requires a broad array of specialized roles. 
	- Evaluation and measurement are crucial for scaling a product beyond vibe checks. The skills for effective evaluation align with some of the strengths typically seen in ML engineers.

Rough progression of the roles you need:
1. First, focus on building a product. This might include an AI engineer, but it doesn't have to.
2. Next, create the foundations by instrumenting your system and collecting data. You might need platform/data engineerings, and systems for querying/analyzing data to debug issues.
3. You will eventually want to optimize your AI system, including designing metrics, building eval systems, running experiments, optimizing RAG retrieval, debugging stochastic systems, and more -- MLEs are really good at this.











