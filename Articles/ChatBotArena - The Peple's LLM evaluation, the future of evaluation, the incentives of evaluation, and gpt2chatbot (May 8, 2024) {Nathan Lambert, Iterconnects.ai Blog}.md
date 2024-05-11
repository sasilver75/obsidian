#article 
Link: https://www.interconnects.ai/p/chatbotarena-the-future-of-llm-evaluation

----

Broadly, LM evaluation these days is reduced to three things:
1. The best academic benchmarks: The [[Massive Multi-task Language Understanding|MMLU]] benchmark, which is used to test general LM *knowledge*, along with some other static/open benchmarks. Developed by everyone, used by everyone, can benefit everyone.
2. A few new-styled benchmarks: The only one to break through here is the [[ChatBotArena]] from LMSYS where state-of-the-art chat models are pitted head to head to see which model is the best. Serves a broader stakeholder audience  by being publicly available.
3. Benchmarks we *don't see*: eg Private A/B testing of different LM endpoints within user applications. The gold standard for evaluation, but you need an existing product pipeline to serve through.

This post focuses on ChatBotArena.

# What is ChatBotArena?
- [[ChatBotArena]] is the side-by-side blind taste test for current language models (LMs) from the LMSYS organization, and it's everyone's public evaluation darling.
- ChatBotArena fills a large niche: "Which chatbot from the many we hear about use is best at handling the general wave of content that most people expect LMs to be useful for?"
	- You enter a question, are given two answers, and then you can vote which of the two answers are better, or continue the conversation. Only once you vote are the identities of the two models revealed. 
	- No sign-ins are required -- anyone can come and go at their liking.
- ChatBotArena thrives in a worldview where LMs are primarily about letting a user get any question answered - where LMs are the general purpose technology. ==In reality, I think **PEOPLE WANT MORE SPECIFIC AND USEFUL EVALUATIONS.**==
- ==ChatBotArena is largely uninterpretable results, providing little feedback for improving models other than tick on the leaderboard for where you stand.==
- When it comes to human evaluation of LMs, it might be interested to look at the [16 page Google Doc](https://docs.google.com/document/d/1MJCqDNjzD04UbcnVZ-LmeXJ04-TKEICDAepXyMCBUb8/edit#heading=h.21o5xkowgmpj) that OpenAI made public about instructions for RLHF grading of [[InstructGPT]] responses.
	- It's known that the rule of thumb is that you can expect to throw out about a third of such collected data (but maybe Scale.ai has gotten better) when using crowd workers... but maybe that's different with OpenAI using experts on everything from math to poetry.
- The leaderboard recently added categories based on simple topic classifiers on the prompt.
	- Excluding Short Queries (often meaningless or vague questions)
	- Excluding ties (artificially suppressing the gaps in the Elo ratings)

# Who needs ChatBotArena? Probably not you!
- ChatbotArena has immense value to all model provider, by allowing them to benchmark their models against other competitors' models on user queries.
- Not everyone training LMs needs this sort of competitive analysis centered around chat -- for many models whose goals isn't *chat*, using this would be comparing apples to oranges.
- ChatBotArena is a living source of ==vibe evals== -- Meta is the pinnacle of a company that should do well there, based on their company values of keeping people engaged, etc. OpenAI is only in the same boat to whatever extent ChatGPT is their long-term company strategy.
- Author thought it was very funny that ==Meta actually got a little pushback for shipping a model that "Gamed" ChatBotArena by having a personality that people like better. Some people have accused Llama of "style doping" in the arena, which Gemeni was also accused of in the past.==
- General evaluations like ChatBotArena is a luxury rather than a need -- good for PR, but not as useful for building something genuinely useful.
- You need a minimum amount of clout or performance in the ML community to even get your model listed.


# Statistics of ChatBotArenas
- ==ChatBotArena has collected less preference data in its existence than Meta bought for [[LLaMA 2]] alone!==
	- The scale by which the leading LM providers operate is much bigger than people think. 
- As of writing, the leaderboard has 92 models and 910k+ votes -- they're now getting 100k+ votes a month now.

> It’s also good to keep in mind that the ELO scores compute the probability of winning, so a 5 point Elo difference essentially still means it’s a 49-51 chance of user preference. Not much better than a coin flip between the first 4 models, especially considering the confidence interval.

The expectations of AI are very different now. **We no longer evaluate LLMs on specific things, we use them for almost all our digital thinking. There is really no solution to all our eval wishes.** Once hoping for the impossible gets old, we will come to expect different things out of evaluation.

LMSYS is becoming an A/B testing laboratory for top labs, which not only dilutes LMSYS’s relevance by including opaque models, but gives laboratories more chances to overfit, and game, the signal available to capture in the Arena.

In summary, I…
- Don’t think prerelease models should be in the arena,
- Wish LMSYS was a non-profit and the amazing students could get more support (having of course been an academic and currently being at a non profit),
- Worry about little things that may dilute the transparency and clarity of the resource.





















