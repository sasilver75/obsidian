#article 
Link: https://gist.github.com/yoavg/59d174608e92e845c8994ac2e234c8a9

-----

A short personal perspective of the author (former Google Research, now lecturer @ a university in Israel I believe) on LMs and where we stand with respect to language understanding.

# Perfect Language Modeling is AI-Complete
- The game of guessing the next token; really playing it at "human level" would mean solving every other problem of AI, and exhibiting human-like intelligence!
- Consider that the game entails completing any text prefix, including very long ones, dialogs, every description of experience that can be expressed in human language, every answer to every question that can be asked on any topic or situation, including advanced mathematics, philosophy, and so on.
	- In short, to play it well, you need to *understand* the text, understand the situation being described in the text, imagine yourself in the situation, and then response -- it really mimicks the human experience and thought.

# Building a LLM won't solve everything/anything
- Yoav used to say: Obtaining the ability of perfect language modeling entails intelligence ("AI-complete"), why did I maintain that building the largest possible language model won't "solve everything"?
- Was he wrong? Sort of
	- He definitely was surprised by the abilities demonstrated by LLMs, and there turned out to be some sort of phase shift between 60B and 175B which made language models super impressive -- they do a lot more than what he thought LMs could ever do -- including many of the things he had in mind when he cockily suggested that they won't solve everything.
- Let's talk about some of the differences that Yoav sees between current-day LMs and what was *back then* perceived to be an LM, and then briefly go through soem of the things that are not yet "Solved" by large LMs.

# Natural vs Curated Language Modeling
- What do we mean when we say "the performance of current day LMs are ==not== obtained by language modeling?"
	- We're talking about ==instructions, code, and RLHF==
	- Training on "text alone" like a "traditional LM" does have some clear theoretical limitations; it doesn't  have a connection to anything "External to the text," and hence cannot have access to "meaning" or to "communicative intent." -- another way to say is that ==when you train on text, the model is "not grounded"== -- the symbols the model operates on are just symbols related to eachother, not grounded to any real-world item.
		- We know the *symbol* "blue," but not the real-world concept behind it.

(Describes IFT, Code data, and RLHF briefly)

# What's still missing?

## Common yet boring arguments
- They're wasteful; training them and using them are very expensive.
	- This is true today, but things get cheaper over time, and the total cost is so far miniscule compared to other energy consumptions that humans do.
- The models encode many biases and stereotypes.
	- Of course! We humans are biased and constantly stereotyping eachother, meaning models trained on our outputs should be subject to scrutiny.
- The models don't "really" understand language
	- Sure. They don't. So what? Let's focus on what they *do* manage to do, and improve where they don't.
- These models *will NEVER really understand language!*
	- There are parts that they clearly cover very well; let's look at those. Those who *really* want to understand language may instead indeed prefer to look elsewhere.
- The models do not understand language like humans
	- Duh, they aren't humans. But they can still te



















