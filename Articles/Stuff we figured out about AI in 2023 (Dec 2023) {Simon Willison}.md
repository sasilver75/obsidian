---
tags:
  - article
---


Link: https://simonwillison.net/2023/Dec/31/ai-in-2023/

----
2023 was a breakthrough year for LLMs! 


- ==Large Language Models==
	- In the last 36 months, we discovered that you can take a GIANT corpus of text, run it through a pile of GPUs, and use it to create a fascinating new kind of software.
	- LLMs can do a lot of things
		- Summarize documents, translate from one language to another, extract information, write code.
		- Cheat at homework, generate fake content, be used for all manner of nefarious purposes.
- ==They're actually quite easy to build==
	- Intuitively, you'd think that it would take millions of lines of complex code -- instead, it just take a few hundred lines of Python to bulid a basic version!
	- What matters most is the training data! You need a LOT of data to make these things work, and the quality and quantity of the training data appears ot be the most important factor.
	- If you have the right data, and can afford to pay for the GPUs to train it, you can build an LLM.
		- The costs used to be significant ($M), but seems to have dropped to tens of thousands, for certain types of LLMs. Microsoft's [[Phi-2]] claims to have used ~14 days on 96 A100 GPUs, which works out to about $35,000 using current Lambda pricing.
		- So training an LLM isn't something that a hobbyist (or many universities) can easily afford, but it's no longer the domain of the super-rich.
- ==You can run LLMs on your own devices==
	- In January of 2023, we thought it would be years before we could run useful LLMs on our own computers -- Then, in Februrary, [[Meta AI Research]] released [[LLaMA]], and a few weeks later, Georgi Gerganov released code that got it working on a MacBook.
	- Later, in July Meta released [[LlLaMA 2]], an improved version that (crucially) included permission for commercial use!
	- Now, we can run a bunch on our laptop -- You can even run models like [[Mistral]] 7B on your iPhone! Or you can run them *entirely in your browser* using WASM, in the latest chrome!
- ==Hobbyists can build their own fine-tuned models==
	- While LLM development from scratch is still out of reach of hobbyists, fine-tuning these models is another matter entirely!
	- There's been a fascinating new ecosystem of training their own models on top of these foundation models -- often publishing those models directly as well as the datasets used to fine-tune them.
	- The [[Hugging Face]] [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) is one place that tracks these -- it moves as a dizzying pace. 
		- Open models have thousands of researchers and hobbyists collaborating and competing to improve them.
- ==We don't yet know how to build GPT-4==
	- Frustratingly, despite the enormous leaps the open source community has had this year, we're yet to see an alternative model that's better than [[GPT-4]], which was released in March 2023. Google's [[Gemeni Ultra]] is soon to be available ot the public, but we don't know if it will beat GPT-4. [[Mistral]]'s team is strong, and their first public model came out only in September 2023 -- will they be able to?
	- It seems that [[OpenAI]] must have some substantial tricks that they haven't shaerd yet.
- ==Vibes-Based Development==
	- As a computer scientist and software engineer, LLMs are *infuriating* -- they're convoluted black boxes -- we continue to have very little idea of what they can do, how exactly they work, and how best to control them.
	- The worst part of the challenge is being able to *evaluate their performance!* There are plenty of benchmarks but no benchmark is going to tell you if an LLM actually "feels" right when you try it for a given task -- you have to directly interact with a model for a substantial amount of time to really get a *feel* for its capabilities!
	- We're left with what amounts to *vibes-based development!*
- ==LLMs are both really smart and really dumb==
	- On one hand, we keep on finding new things that LLMs can do that we didn't expect, and that the people who trained the models didn't expect either -- that's usually really fun!
	- But on the other hand, the things that you have to do to get a model to behave are often *incredibly dumb!*
		- It seemed like ChatGPT might have gotten lazy in December, because its hidden system prompt might have included the current date, and its training data shows that people provide less useful answers coming up to the holidays! Maybe!
			- Cash tips for answers, encouraging the model, telling it your career depends on it, etc. It's all so dumb, but it works!
- ==Gullibility is the biggest unsolved problem==
	- Language models are *gullible* -- they believe whatever we tell them! We need to force them to NOT be gullible!
	- We want them to be (eg) good assistants -- and good assistants don't just believe whatever anyone tells them!
	- If we want to build "AI Agents" that can go out into the world and act on our behalves (this is a vague term)... they can't be gullible. 
	- Can we solve this? Simon is beginning to expect that we can't until we achieve AGI. So it might be a while before those agent dreams can really start to come true!
- ==Code may be the best application==
	- Over the course of the year, it's become increasingly clear that writing code is one of the things that LLMs are MOST capable of!
	- If you think about what they do, this isn't such a big surprise. The grammar rules of Python and JS are MASSIVELY less complicated than they are in English.
	- The thing about tocde is that *we can run generated code to see if it's correct!* * And with patterns like [[ChatGPT]]'s Code Interpreter, the LLM can execute he code itself, process the error message, and then rewrite it and keep trying until it works! Cool, right!
		- So in this sense, ==hallucination is a much lesser problem for code generation than it is for anything else! We can at least partially test the "truthiness" (run-ability) of code, unlike English!==
	- How should we feel about this as software engineers? It feels as a threat, in one hand, but on the other hand, as software engineers, we're better-placed to take advantage of this than anyone else!
		- We've all been given weird coding interns! We can use our deep knowledge to prompt them to solve coding problems for effectively than anyone else can!
- ==The ethics of this space remain diabolically complex==
	- For generative art, there's the issue of how Stable Diffusion acquired unlicensed training data.
	- Since then, almost every major LLM have also been trained on unlicensed training data!
		- The NYT launched a landmark lawsuit against OpenAI and Microsoft over the issue. The 69 page [PDF](https://nytco-assets.nytimes.com/2023/12/NYT_Complaint_Dec2023.pdf) is genuinely worth reading -- especially the first few pages, which lay out the issues in a way that's surprisingly easy to follow. The rest of the document includes some of the clearest explanations of what LLMs are, how they work, and how they're built, that Simon's read anywhere!
	- The legal arguments are complex -- Simon expects the case to have a profound impact on how this technology develops in the future.
	- Law is not ethics. Is it Okay to train models on people's content without their permission, if those models will then be used in ways that compete with these people? {What if these same models will also be used in Medical/Therapy applications?}}















