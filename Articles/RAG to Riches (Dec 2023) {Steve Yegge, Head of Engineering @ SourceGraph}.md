Link: https://sourcegraph.com/blog/rag-to-riches
Note: This isn't that interesting of an article, it's mostly a news roundup. But it's written in a fun-to-read style, which is a good takeaway.

----

News Story 1: AI Grows Up

If you've spent more than roughly 257 nanoseconds online in the past 3 months, you'll have noticed that the AI competition has become heated -- from new companies saying they're going to topple the incumbents with AI, to big companies saying they've been doing AI secretly all along and not telling anyone.

It's more than a fad -- we're in the "craze phase," where putting AI in your company's name gets VCs to just blatantly throw money your way.

At SourceGraph AI, we're paying close attention to the space (dominated by [[OpenAI]]). The big news is that [[Anthropic]] is giving competition, and others are starting to get serious.

---

News Story 2: Bear Spray Engineers

You've no doubt noticed that GPT has spawned an ecosystem of hundreds of makeshift AI assistants that swarm you like gnats. They all claim to automate every kind of daily drudgery in your life, from summarizing your Slack threads, emails, newsfeed, etc.

But one AI-assistant category seems to have made real, tangible products -- ==coding assistants==! Developers everywhere will readily agree that coding assistants are hyped to the pointer where your IDE will helpfully prompt you to install one roughly every 40 seconds... and you'd gradually pay the subscription fee just so that it'd stop asking!

And yet strangely, only about 1-2% of all GitHub monthly active users pay for Copilot! What are the other 98-99% of you doing?

Probably looking for a job -- Every company is writing in an AI-induced Innovator's Dilemma, which made everyone's jobs harder, and there are tons of folks who are gunning to move. The market is flooded with talent, and for various reasons, every company is only hiring Security engineers and ML engineers! So almost nobody else CAN move!

For those not in the know, an ==ML engineer== is like a regular engineer, but they're driving a Ferrari. They're adept with ML infrastructure, designs, tools, practices, and deployments. If you're an ML engineer, then you've by this time already figured out that you need to carry bear spray around to fend off recruiters who appear out of nowhere and hurl heavy, dangerous wads of cash at you.

-----

News Story 3: AI Strategies are Beginning to Emerge, and in Some Cases, Prolapse

Everyone talks tough, but most companies are in the very earliest stages of figuring out their AI strategy. At Sourcegraph AI, we talk to many companies, and have the same conversations over and over.

There are several distinct profiles with respect to AI strategy that we can bin companies into:
1. The rarified few who turn in their homework ahead of time and do extra credit -- looking at you, CapOne.
2. The ones whose homework is kind of on track, but who aren't really sure how to bring it to the finish line -- this is about 30% of all companies.
3. The ones who, when we ask about how their AI evaluation homework is going, hardly know what we talk about. This is 50% of the Fortune 1000.

"Homework" here means figuring out your AI strategy -- and this is homework that every company on earth needs to be doing right now. And yet, many are handling planning with very little urgency.

In your industry, there are at least one or two snotty straight-A students who are putting together five- and ten-year plans for their AI homework -- and many of them are specifically well-down the path of evaluating coding assistants for their developers.

The ==Innovator's Dilemma==, which is more like "torture device" than "dilemma" is tearing companies up, and keeping them from responding to the new pressure from AI.

The  foot-draggers say: "We're working on it -- we just have to finish a reorg, and so-and-so just left, and we have a task force that we're trying to put together but things are just really chaotic" -- they just can't seem to get *organized* around AI -- it's not jelling, for some reason!

For most of you, that reason is the Innovator's Dilemma -- and your company will have to develop a sense of life-or-death urgency to get through it. Don't get left behind by competitors who are getting huge speedups from AI in general, and from coding assistants specifically.

----

*News Story 4: Productivity Metrics are Smelling Gamey*

- The newfound and unprecedented productivity boost from coding assistants has awaked an alarming new idea in many companies -- which is that they need to try to *measure the improvement in productivity!*
- You can see where this is going. Many customer ask Sourcegraph if Cody can be modified to have a detector to monitor how productive their engineers are, with metrics to detect when they fall below a certain threshold.
- Discerning companies have converged and fixated on what is now the most populated coding-assistant metric, [[Completion Acceptance Rate]] ([[Completion Acceptance Rate|CAR]]), which measures how many times you thought -- "well, that code looks kinda right" and hit TAB to accept it, out of the total number of completion suggestions you're shown.

----

News Story 5 - No comment

- Steve is a huge proponent of commenting your code, and doesn't understand people who are against it. Luckily, it turns out that ==comments provide incredibly valuable context to help LLMs reason about your code!==
- During a query, an LLM might only be looking at a very small hunk of code from a larger subsystem or scope, since you have to often cram many code hunks into the context window to craft the best context.
- Comments in those code hunks not only help the LLM reason about your code with broader and more accurate context, but Cody can also use them upstream in the context assembly engine for augmenting similarity search, graph search, and other indices.
	- Comments make the whole system smarter.

----

News Story 6: Cody Grows up and Goes GA. Already.

- Cody, Sourcegraph's LLM-backed RAG-based AI coding assistant, is now Generally Available - for you!
- You can use Cody in several clients, including the flagship VSCode (GA), IntelliJ (Beta), other Jetbrain IDEs (Experimental) and Neovim (Beta). And it has a free tier to just screw around with it.
- The completion suggestions are fast and satisfyingly accurate, and the Cody chat experience is basically unparalleled, with the chat UI and context specification being dramatically improved. 
- Cody's secret sauce is that Cody is the perfect [[Retrieval-Augmented Generation|RAG]]-based coding assistant, which is by definition one that produces the best context for the LLM to process.
- This is what RAG is all about -- you *augment* the LLM's generation by retrieving as much information as a *human* might need in order to perform some task, and then feeding that to the LLM along with the task instructions.
- Cody's RAG is a variant of a recommendation system -- recommending relevant code and other context to the LLM for each task or query.
	- Producing the perfect context is a dark art today -- Cody is likely the further along here from what the author can see.
	- =="The key to generating great context is to look at the problem from many different lenses, and each of Cody's backend context providers is a different lens."==
- On the retrieval side, Cody has many code-understanding lenses -- among them, a high-speed IDE-like code graph, multiple high-performance code search indexes and engines, natural language search, and more -- all acting in concert as specialized retrieval backends.
- On the context *fetching* side, for getting information for Cody to slice/dice -- ==Cody is beginning to incorporate multiple context sources *beyond* just your repos==! This gives Cody lenses into metadata and documentation that's not necessarily evident from the code itself -- such as any really-critical important information that the author of the code *didn't* put into a comment (eg documentation).
- 



