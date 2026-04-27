From [[Steve Yegge]]

Gas City is a ground-up rewrite of [[Gas Town]] from first principles using the MEOW stack (i,.e. [[Beads]] and [[Dolt]], the fundamental substrate of Gas Town).

It is almost a perfect proper superset of Gas Town's features (and can be a drop-in replacement), but Gas City is *also* an orchestrator builder.
- Gas Town is a "pack" within Gas City, a fully declarative bundle of prompts and skills, no code at all.
	- It still has the iconic original Mayor, Deacon, Dogs, Polecats, Witness, Refinery, and Crew, with their various hooks, inboxes, skills, prompts, and sandboxes.

Gas City was built by his friends Julian Knutsen and Chris Sells, and as far as Steve can tell, it's exactly what he envisioned and outlined to them when they first suggested tackling it. 

It's so good that Gas Town itself, the binary, is not long for this Earth.

==If you think of Gas Town as a Dark FActory, then Gas City is a Dark Factory Factory==.

Gas City has ==deconstructed the entire Gas Town stack into composable, declarative building blocks== called ==“packs”==. You can use these to assemble arbitrary agent topologies, deploy them, sit back, and watch them work from a rich console.

As its unboxing flex, Gas City ==comes with a fully functional “Gas Town” pack==, which runs an exact replica of Gas Town. This is the default pack that runs on startup. So Gas City starts off as a drop-in replacement for the original Gas Town, and can import all your rigs and beads.

 It’s Beads all the way down, powered at the base of the stack by a unique git-versioned database called Dolt. Dolt was the magic that made our stack run smoothly.

Gas City solves an enormous number of problems associated with spinning up long-running agentic worker “teams.” It builds atop the innovations and community contributions from Gas Town, giving you out-of-the-box, scalable, convenient access to agent identity, messaging, history, context, state, skills, roles, personas, and much more. And for coding agent maintainers, Gas City ==exposes a rich Factory Worker API==. It’s a way to make your own agent the driver for Gas City.

Gas City is generally an improvement on Gas Town on all fronts, from code quality, to the services it offers. For instance it ==offers fine-grained model selection and switching at various levels, for cost control==.

Gas City doesn’t aim to solve all your problems. ==You will need to wire it to your own sandboxing, MCP servers, and so on.==

This combination of tech stack and community makes Gas City, as far as I can tell, the only viable solution for building custom orchestrators backed by Git. You can build and run an entire business with it, tracking every step taken by any agent in a database with git version history. The forensics and auditing capabilities of Gas City are unparalleled, because of MEOW and Dolt.


# The Light Factory

Gas City, like Gas Town, has chosen to maximize Observability. You can dive in and interact with any worker at any time, and nothing is ever hidden from you, nor from the agents, except for the guardrails you choose to install.

Claude Code did not start life as a dark factory, because you were generally supposed to watch it while it works. It’s making overtures in that direction with subagents and agent teams, but they keep the lights off intentionally, presumably because it’s a consumer-facing product and it keeps it simpler.

Gas City takes a different approach: all agent workers are equally visible and addressable. The lights are on. Normally for the ephemeral “polecat” workers, you don’t bother looking at them. But unlike with coding-agent subagents, if you want to talk to your polecats, you absolutely can.

And so far, Gas City is the only dark factory that has been designed with the goal of creating other factories. The lights are 100% on when you’re working with the Mayor and Crew in Gas City, and you can dial them up as needed in the back rooms with the polecats and dogs. For this reason I have begun to think of Gas City as a Light Factory… or at least, a very well-lit dark one!

==You should almost never deploy a single-agent pack for a real business process.== The reality is that any agent can go temporarily insane, at any time, and make a bad call. No matter how smart they are. We know now that hallucinations and false memories and forgetting are baked mathematically into all memory systems; there’s no avoiding it. ==So you should never just have one coding agent managing a piece of infrastructure. Not even for a low-stakes part of your business==. You should always have at least two or three working together on a little crew.

==This is exactly why dark factories are so attractive. With Gas City you can build any sort of adversarial group structure you like, for a team of collaborating agents. They can watch over each other==. By catching each other’s mistakes, the agent group reaches a far more reliable consensus and outcomes than you can get from a single agent. That’s why we think of deployed orchestration as being fundamentally made of multi-agent teams: factories. Define your pack, deploy it, et voila — you are officially on the path to being an AI-native shop.


![[Pasted image 20260427010142.png]]


  
Reliability, friends, is a dial. You choose where to set it. More rounds of review, more backstops, more guardrails, more judges, and you can get agentic workers to be as reliable as you need them to be, at least up to some practical ceiling. I wouldn’t use it in situations where you could physically hurt people, e.g. in medical or navigation systems. Not in 2026. But we’ll build our way there, like engineers do, over the next couple of years.

