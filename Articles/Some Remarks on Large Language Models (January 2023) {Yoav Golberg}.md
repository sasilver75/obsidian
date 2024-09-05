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
	- Duh, they aren't humans. But they can still tell us a lot about language structure.
- ...
- The models don't cite their sources
	- Indeed they don't, but... so what?
	- I can see why you'd want that in certain situations, and you certainly want the mdoels to not bullshit you, but these are all not really related to the core of what language models are -- the author doesn't think this is the right question to ask -- after all, humans don't "cite our sources" in the real sense -- we rarely attribute our knowledge to a specific single source, and, if we do, we often do it as a rationalization.

## So what's missing? What are some real limitations?
An informal and incomplete list of things that are currently challenging in LLMs, including the latest ChatGPT, which hinders them from "fully understanding" language in some sense.

- Relating multiple texts to each other
	- In training, models consume text as a large stream, or as independent pieces of information. They might pick up commonalities in the text, but it has no notion of how the texts relate to "events" in the real world. 
	- In particular, if the model is trained on multiple news stories about the same event, it has no way of knowing that these texts all describe the same thing, and it cannot differentiate it from several texts that describe similar to but unrelated events.
	- In this sense, model's can't really form a coherent and complete world view from the text they "read".

- A notion of time
	- Similarly, models don't have a notion of which events *follow* other events in their training system; they don't really have a notion of time at all, maybe besides explicit mentions of time -- so it might learn the local meaning of expressions like: "Obama became presidation 2009" and then "reason" about other explicitly-dated things that happened before or after thins, but it can't understand the flow of time in the sense that if it reads that "Obama is the current president of the US" and then "Obama is no longer the president," it might concurrently "believe" that Obama is the current president of the US _and_ Trump is the current president in the US _and_ Biden is the current president of the US.

- Knowledge of Knowledge
	- The models don't really know what they know. 
	- The model's training and training data don't have a way of distinguishing between predicting tokens on well-founded, acquired knowledge, or on a complete guess.
	- The model's training and training data doesn't have an explicit mechanism for distinguishing these cases, and certainly don't have explicit mechanisms to act differently according to them.
	- RLHF made the models "aware" that some topics should be treated with caution, and maybe even models learned to associate this level of caution with the extent to which some fact/entity/topic were covered in their training data, or the extent to which the data is reflected in their internal weights... So they have *some* knowledge of knowledge
		- But they often get over this initial refusal-to-answer stage, and then go into "text generation mode".

- Numbers and math
	- The models are really ill-equipped to perform math.
	- Their basic building blocks are "word pieces," which don't really correspond to numbers in any convenient base.
	- There are much better ways to go about representing numbers and math than the mechanisms we give LLMs -- it's surprising they can do anything at all.

- Rare events, high recall setups, high coverage setups
	- The models focus on the common and probable cases
	- This makes me immediately suspicious about their ability to learn from rare events in the data, or to recall rare occurrences.

- Data hunger
	- The biggest technical issue with LLMs is how data hungry (trillions of tokens) they are.
		- Most human languages don't have so much data -- certainly not in digital form.

- Modularity
	- At the end of the "common yet boring arguments" section above, we ask: "How do we separate "core" knowledge about language and reasoning, from specific factual knowledge about "things?"
	- This is a major question to ask, and solving it will go a long way towards making progress on other issues.
	- If we can modularize and separate the "core language understanding and reasoning" component from the "knowledge" component, we might be able to do much better WRT the data hunger problem.



















