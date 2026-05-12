



Q: Why [[Elixir]]?
A: The process supervision and the gen servers are super amenable to the type of process orchsetration we're doing here
- You'er essentially spinning up little daemons for each task that is in execution, and riving it to completing, which means the model gets a ton of stuff for free by using Elixir and the [[Bogdan Erlang Abstract Machine|BEAM]])
- Most people are not operating at the scale where you need this, but it's a good mental model of resumability and all those things...

Q: Tell me about the origins of Symphony?
A: At the end of december, we were at 3.5 PRs (when GPT 5.2 was out) a day.
- With 5.2, we were up in the 5-10 PRs per day per engineer. This becomes very taxing to constantly being switching like that; we were pretty tapped out at the end of the day.
- ==Where are we spending time?== Humans were spending their time context switching between all of these active tmux panes to drive the agent forward.
- So we wanted to build something to remove ourselves from the loop, so we sprinted after a way to find a way to remove the need for the human to sit in front of theri terminal.
	- Lots of experiments of dev boxes, spinning up agents, etc... it seems like a lovely world of sitting on the beach, etc... 
	- Its' a super interesting framework for how the work is done:
		- I'm less latency sensitive
		- I'm less attached to how the code is actually written; if the code is garbage, I can throw it away and not care about it.

In symphony ther'es a rewrok state where when aPR is proposed and escalated to a human for review:
- It should be a cheap review, mergable or not
- If it's not, move it to rework state, and the elixir service will track the entire work tree and PR and start it from scratch.
	- "Why was it trash? What did the agent do that was bad? Fix that before moving into progress"

Q: You guys are ahead of codex app
A: the team has been as AI pilled as possible and sprint ahead, and a lot of the things that we have worked on have falled out into a lot of the products that we have. We wre in deep consultation with te codex team to have the codex app be a ting that exists, to have skills be a thing that exist, to put automations into the product, so that all of our automated refactoring agents didn't have to be handcrafted control loops.
- It's been nice to very quickly try to figure out what works and later find the scalable thing that can be deployed widely.
- It's been fun and chaotic. I've often lost track of what the state of the code is.
- ==We've wired Playwright up to the Electron app with MCP. MCPs... I'm pretty bearish on, because the harness forcibly injects thsoe tokens in teh context, and I don't have a say with it. They mess with autocompaction, the agent can forget how to use the tool.... ther's probably onl ever like 3 calls in playwright that I ever want to use, so I pay a big cost for a few things==
	- ==Someone vibed a local Daemon that boots Playwright and exposes a tiny little shim CLI that drives it, and I had no idea that this occurred!==