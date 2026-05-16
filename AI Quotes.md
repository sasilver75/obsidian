
You want tools for the agent to be able to close the loop for itself (and understand that it's right), and you also want it to make it easy for you as the human to verify that the result is correct.

"Spatial isn't Special," or rather it shouldn't be.

Everyone is running around looking to go 15-30x faster right now, and I think people should be trying to figure out how to reliably do 2x, with foresight and planning. 

I think people are going to realize that they've stopped reading the code, and it's causing issues. Amazon had an outage where they had a SEV0 and had a 90 day reset where juniors and mids couldn't review eachother's slop PRs, and everything had to be reviewed by seniors. We are back to reading code. The idea that it could take days or weeks to fix a bug could be a company-killer, depending on your company, if you haven't looked at the code for 3 months and your automatically-launched bug resolution agent isn't able to solve the problem.

It's hard to make models that are good at software architecture from an RL perspective, I think, because whether an architecture is good or bad is something that plays out over years as it evolves organically. It's an interesting space.

People say "Let's write slop code and GPT-7 will clean stuff up." Everyone's waiting to get their hands on things like Mythos, we'll see. When you say "I don't need to read my code," you're making a bet about when that model is going to come out. I wouldn't bet my future or my company's future on that. You might want to go a little slower and guarantee that no matter when that model comes out, you're good. You should skate to where the puck is going, but you shouldn't put everything on the line. Build for the model 6 months from now, but have moderate expectations about it. It might (~5%) be a 2x model, but it's more likely to be a +20% model.

Some F500 companies are excited to have just onboard Microsoft Copilot for every employee, and some people in SF are running many-dozen agent swarms, trying to burn as many tokens as possible.

Labs are saying "software engineering is solved," but many companies have adopted claude code and seen 15%-30% or so improvements in productivity.

If you sit there and YOLO prompts into the model, you will get results... but if you really understand the models, you can push them to solve things faster at higher quality.

Dex Horothy recommends if doing complex work, having max two things at a time, and always having a top priority. There's a primary thing, as soon as it's unblocked, it becomes the thing he's working on. It's two hard things or one hard thing and 3 small bugs that you don't have to worry about. People are running around trying to get 10x speed... but planning, research, etc... each give you 2x, 3x leverage; learn to do that well, and then go figure out how to do 10-30x faster. Just learn to get to 2x faster, and it's transformative to you, everyone else on your team, your company, etc. Stop trying to moonshot everything.

_______

Specs to Code
To build an app, the best way is to take some sort of specification document (PRD) and turn it into code. If there's something wrong with the r resulting code, you don't look at the code, you look back at the specs. This is vibe-coding by another name, in the sense that you keep editing the specs... Pocock tried this, and it sucks; it doesn't work. You need to understand the code, and shape it.

________

Steve Yegge

Token burn as a metric... what it says is that your engineers are trying. And that they're trying, they're failing, and they're learningIf you want to find where your bottlenecks are, you need to start now, which means try. It doesn't matter what tool you use, it matters that you're learning. The companies that win are the ones that experiment the most.

There's a huge problem with people thinking it's garbage; you pick up the shovel and dig with it; you can't just say "shovel, dig!" Most people can't read. Much of Steve's work... has gone down the wrong path by overestimating peoples' ability to read. CC makes you read a LOT. Until the UIs that arrive that are good enough for everyone that can't read... those people will be at a severe disadvantage.


https://youtu.be/aFsAOu2bgFk?t=3780
The heresey here is a very important idea.

Forking used to be a declaration of war. I think now it's going to be an everyday thing, because open source projects aren't really accepting slop PRs right now. So just take it and remix it.

_________


￼

___________

https://youtu.be/dHBEQ-Ryo24?t=2073
A lot of organizations are betting their future on a speculative premise that AI is going to be able to do everything (or everything in coding) better than humans. I worry about this a lot for both the organizations and humans. For the humans, if you don't use your engineering muscles, you don't grow, you might even wither. 
As the CEO of an R&D startup, if my staff aren't growing, we're going to fail. Getting better at the particular prompting skills, whatever details of the current generation of AI, CLI frameworks... isn't growing. That's as helpful as learning bout the details of some AWS API when you don't understand how the internet works. It's not reusable knowledge, it's ephemeral knowledge. If you want to, you can use it as a learning superpower, but it can also do the opposite. The natural thing it does is remove your competence over time.
"How AI impacts Skill Formation" from paper; most people didn't learn. The ideal situation for GenAI coding is that ... we can stay in touch with the process, using our domain and business context knowledge, etc. But the default attractor is for people to go into an autopilot mode, and they have no idea what's happening, and it's making them dumber.
The people who are benefitting from AI are really junior people... and really experienced ed people, because we can have it do some of our typing for us. People in the middle, most people... it really worries Jeremy, because how do you get from point A to point B without typing code... It's kind of like going back to school, where we don't let students use calculators so they develop their number muscle... do we do that for developers? I don't know. If I was a 2-20 years experience developer, I'd be asking my question that a lot, or you'll be in the process of making yourself obsolete. 

I've never felt more drained writing code agentically. We've spend the last couple of years building a product around this.

____________________

Steve Yegge

Everything they do, assume that it's 85% correct.

Everyone's chasing the same form factor, Claude Code, Amp, Codex, Gemini CLI, etc. All of them came out in February. The ergonomics matter a lot. It's like a car, you spend all day in it.


___________________

Steve Yegge + Gene Kim (authors of Vibe Coding)
￼
￼
- Coding agents are manual tools, like drills or forklifts
- You can be very skilled with it, and do precision work, or cut your foot off..
Next year, we will move from saws and drills to CNC machines.

People say: "The models have plateaued," even if they did, we have still discovered steam and electricity, and it just takes some engineering for us to harness it. All code in 1-1.5 year will be written by giant, grinding machines. 

￼
Some percentage of their engineers are using Codex, and a larger percentage are NOT using Codex, and the difference in productivity is staggering, because the difference in productivity is staggering.
- They have to fire half of their engineers, which is unfolding at other companies too.
- Senior and staff engineers are the ones refusing it.

this is just like what happened in the swiss mechanical watch industry, and how Quartz killed it in a couple years.
- The craftsmen did the same thing our staff engineers do today (violent protesting)

We've established that CLI agents are too hard for most devs; We clearly need a UI. We also need to lift people up (CC shoves your nose in everything, you have to watch everything it does. We need helpers, models!)

You context window is like an oxygen tank. You're sending a diver down in your code base to fix stuff for you. They're saying "We're gonna give me a bigger tank, 1 million tokens!" He's still gonna run out of oxygen.
- You should send a PM diver, then a coder diver, then a test diver, etc... but not one's doing this, everyone's building a bigger diver.

As engineers, task decomposition, successive refinement, modularity, black boxes, components.... we will build an AI colony with lots of agents, not just one agent.

Until then, learn CC, give up your IDE. Hot Take: If you're using an IDE ... you're a bad engineer.

￼
The backlash is very real against this. 
This is 60% of the engineers in your org.

==Referring to the "mechanical advantage" that these tools give us==

￼



_________

https://youtu.be/zuJyJP517Uw (Dec 26, 2025)

Merging is the WALL that everyone building agentic orchestration systems is trying to solve right now.
- I think Graphite has the best shot

When you get to the point that every developer is 10x productive, merging becomes a terrible problem. Engineers A and B both make 20,000 line changes, and A merges his, then B comes along, and A has changed the logging system, architecture, APIs, etc that you were using! You're gonna have to reinvision, reimagine, reimplement your change on my change! 

The important thing is that they have to be serialized... it's a queue, and when they go in there, they have to redo what they were doing on top of their new thing.

No one's solved it, and it's a HUGE obstacle right now.

One company said: "Heres our solution! One engineer per repo!"

swyx: The classic solution for this is stacked diffs, merge queues...
steve: idk anything about that lol. With the agent mail thing... the agents might be able. to figure it out. 

swyx: What do you and Jeffrey Emmanuel disagree on?
steve: Whether 12 agents working in a single repo clone is a good idea. I like separate worktrees with lots of branches or separate repo clones, sandboxed. Jeffrey has them all in the same git, the same build... so one of them will be running a test, etc... 
swyx: That's so much churn!
steve: yeah, he has some file reservation system. He's a solo dev, and he uses a similar principle to Beads. You tell AI "If anything gets messed up, just fix it!"


Swyx: Some people have proposed that the theme of this c
Steve: We're in the phase where we're cutting down corn with scythes right now. We're moving to machines that churn... giant ones you see on factory farms. We're going to be factory-farming code. A lot of people are so dead-set against that morally/philosophically.
Swyx: They're so used to sophisticated agriculture.
Steve: We're moving into the John Deere area of coding. The idea that CC, Amp, Codex... they're all equally bad. They're like a powersaw or a powerdrill; a skilled craftsman can do a lot of good with them, but you can also cut your foot off with them. Imagine a big farming machine that knows how to RUN claude code: "You, plan! You, implement, You, test!" People are building it right now. It's started to unlock programming for non-programmers, which is totally flipping upside-down. Coding is no longer the bottleneck, the business needs to get immediately involved. It's really exciting times, but it's too much for people, and they're either checking out, or revolting online. We will see a massive backlash from... the "luddites."

Swyx: A lot of people in our audience are critical of going whole-hog with this; fine for frontend, fine for application code, but don't touch my cloud infra, my backend, my distributed...
Steve: Don't touch anything production, only touch code, where git is your backstop. If you have git as your backstop, why would you be worried?
Swyx: People have the perception that it's not as good at backend code. 
Steve: The misunderstanding here is rooted in a fundamental belief that the models are done getting smarter... They could NOT, and we'd still be over the hump, where we've discovered gas/steam/electricity, and just need to harness it. 
There's an interesting tension where you're building tools for capabilities that the models will eventually have built into their brains, so there's this constant arms race and decay of your tool filling gaps for the model until the model is good enough to fill it itself, and moves on. All code and tools are becoming throwaway, and they're easier to build too. Joel Spolsky gave an old tech talk... 20 years ago, which was "Never rewrite your code." We've discovered for larger and larger bodies of code, that it's better to start over and rewrite it from scratch than fix it. I've noticed this when porting all unit tests from one architecture to another... if you just say "Throw all the tests out and make them again," it brrrr zips through hem all. I feel like I'm in upside-down land. you have to embrace this new world.
Swyx: A young kid could say what you're saying and not be as believable, but you've done it all.

Swyx: One thing to get your comment on is Google...
Steve: All of OpenAI, Anthropic, Google are chaos, internally. Anthropic hides it best, but they're hiring hundreds of people for CC. You're going to have chaos. Eventually, you're going to have some forming. Google's so siloed, it's a billion monoliths, so it's hard to roll things out across google. All three have execution problems right now. Anthropic's probably doing the best.
Swyx: FB is having problems...
Steve: As soon as open models get as good as Claude Opus 3.7... that's going to be a big deal. They're about 7m behind, and that gap may be gradually narrowing... 

Steve: About the "knowing how to code" being an advantage..
Steve: You have to know functions, classes, objects, etc... and from thereup... you care about how it works, so you've reached the level about how things work architecturally... "Cloudflare does this," "Apache Cassandra does that," ... You still have to learn all this. You still have to learn a massive amount of stuff to be an effective engineer in the new world.

Swyx: What about low-status, high-status jobs... 
Steve: F1 car drivers don't know how to build a car, but they know more about operating it than the people who build it, and so they have to have this conversation... but in the F1 movie, they call them monkey.

Swyx: Tech is fun again. Tech got boring for a little bit. For a while, it was like... "Sourcegraph indexes your codebase very well, and that's cool... but that's not FUN."

_____________

Tim O'Reilly and Steve Yegge conversation: https://www.youtube.com/watch?v=CQKuitliNmc 

You're exhausted after a few hours of doing this because AI is taking all of the easy decision shlep work away from you, and left you with the hard work. It's you're biking and it's all hills.

There's a big gap between being able to invent something, and then all the hard work about getting it out into the world. Not everything is solvable by your AI army. This is a big takeaway: There's always more work! It doesn't matter how superhumanly good your AI is. Our ambition will always outstrip our compute.

Q: Why does Gas Town converge to hierarchy?
A: If you look @ Wasteland, it's an explicit thing that federation is the way, not hierarchy. We modeled it on the construction industry. The construction industry has worked from 1000s of years without a deep hierarchy; you just find someone to build something for you. In Gastown, there's some hierarchy, because it helps to organize teams.


It's not just a matter of "Oh, what do I do?" People are grieving! There's a lot of ambiguity, change, and open-endedness. So there's a lot of leadership that needs to be done, with empathy. 


You can refer to "Precedence" or "Discovery" as a lawyer; "What is the language that calls out from the LLM, the behavior that you want?" Every profession is basically a specialized dialect with which people can compressed to eachotehr with a compressed context. 

Tim O'Reilly: Re: AI use atrophying important pathways; Socrates thoguht that writing would atrophy our critical ability to remember, nad he was rigt! People used to beable to recite long poetry! Things will change, and you can atrophy is you don't actually use your mind... there's a poem from Aroqua (?) about an old testament storty of Jacob wrestling with an Angel: He can't win, but he knows that he comes away stronger from the fight. "What we fight with is so small, and when we win, it makes us small. What we want is to be defeated decisively by successively greater beings." If you're letting your thhinking atrophy, it's because you're not trying hard enough to find things to wrestle with!
Steve: We're gonna build bigger stuff, and it's gonna be fun.


_______________


A machine cannot take responsibility. I don't really see the future where we can automate the software creation process such to the extent that some human won't be responsible.


-------------------

https://www.youtube.com/watch?v=CeOXx-XTYek
Harness Engineering talk 

Step back, take a systems thinking approach. Where is the agent making mistakes, where am I spending my time, how can I NOT spend my time there going forward, and put automation there in place so that you can solve that part of the SDLC.

________________

https://youtu.be/am_oeAoUhew

Each engineer in this room has access to 5, 50, or 5,000 engineers worth of capacity, 24/7, every day of the year. Our role is to figure out how to productively deploy these resources. In this world skillsets are shifting towards systems thinking, system design, and delegation, to make use of this abundant capacity in order to solve problems.

As Late 2025,
1. The models are good enough (e.g. GPT 5.2+) to solve real problems in real codebases.
2. Code is free. Code carries maintenance burden, but it's free to produce, free to refactor, and not something to get hung up on anymore. Models are incredibly patient and infinitely parallel... the ability to produce, maintain, refactor, and delete code is no longer a forcing function on figuring out how to allocate resources on your engineering teams. 
3. Your role is to unblock your team. It's your role as software engineers to figure out how to unblock your team of agents, and humans driving those agents, from being able to drive them over long-horizon work to do the full job. Every one of you is a staff engineer, and you have as many team members as you can drive. Look 1 day, 1 week, 6 months into the future to figure out what structures you need to put in place to harness this incredible capacity to produce code.


The strict resources:
1. Human time. Figure out where this human time is going, figure out ways to productively automate it, and move that synchronous time into higher-level activities.
2. Human model and attention. In a world where program time is scarce, we have P0s, P1, and P2s are never done. But now, P3s get kicked off immediately, maybe 4x in parallel, we choose the best one, and off it goes. When code is free, internal tools can have good localization and internationalization from day one. I can make tools that colleagues in London/Paris/Zurich/Munich are able to experience in their native language. The best parts of SWE are available in every product that we build, all the time.
3. Model context window. The important thing is not the code, it's the prompt and guardrails that got you there. Leaving breadcrumbs, documentation, ADRs, persona-oriented documentation around what a good job looks like, all the historical logs of ticket and code reviews. This is the process that got you and your teams to the products you have today, and this is what needs to happen to get your agents there as well. Your job is to build systems, software, structures that enable your team to be successful.
    1. To do this, we need to make them legible to the agents that are driving the implementation. This means structuring them in a way hat's native to agents, writing them in a way hat's respectful of scarce context, and figuring out ways to make the tokens that are required to do the job easy to predict. making things the same, as much as possible, to limit the amount of attention the model needs to activate to do the job.
    2. Large scale refactoring is free, so making things the same is something you can easily do.


It's our job to specify those non-functional requirements, and to write them down in a way that the agents can see. If the agent's aren't doing that, it's our job to refine and restrict their output such that the code they write is acceptable.
- Say: "Do not produce slop.", but to do that requires taking short-term velocity hits to back up or double-click into a task to figure out what the agent is struggling with in *your environment,* put the guardrails in place to help them, and then step back and try to solve higher leverage things.
- ==To operate in this way, we need to step back and look at the durable classes of failures that the agents and humans in the codebase are making time after time, figure out why we're spending time on it, devices a solution to systematically eliminate this class of behavior, and then continue to observe, refine, and make choices on these non-functional requirements.==
- So my behavior can move up to think about differences in model behavior between releases rather than deeply understanding the nuts and bolts of the harness; how do I drive the behavior I want based on the observed behavior rather than the inner mechanics of the thing.


Q: Human-Agent Collaboration Tool
a: It's largely been MD files in the repo and Github...
- If you think about collaborating on a document (e.g. Google docs. write something, ask for feedback, apply suggsetions), this is a clean room environment jus for this work artifact you're producing (a PR, e.g.)
- All of the agent and humans collaborate together... in a PR (?). We don't block on contribution ... the implementation agent can acnkowledge, defer, or reject any feedback it gets, allowing each participant in the production of diffs to make their own judgements around what it means to deliver, receive, and respond to feedback.
- Has the nice property of not putting het model in a box, in a bunch of places. Being sueper prescriptive about what sorts of feedback must be addressed... can have a catastrophic failure mode of your implementing agent being bullied by all of the reviewers, when really we want to bias towards being accepted, not perfect, not drowning with minutiae.


Q: How should people get started using coding agents?
A:
- Two ways to approach
	- Start using the coding agents to improve your confidence in the code itself as it's written today.
		- More tests is probably a god thing; to assert that our programs are well-specified and behave correctyl as agents interact with them.
		- Agents are good at looking at existing code with some context around how it meant to be used, and writing tests that assert that behavior.
		- Using this to improve your confidence in the code will also improve the agent's ability to navigate around...
	- Look at how you're spending your time: Is it staring at editor writing code? Is it waiting for tests? Is it waiting for human feedback? Is CI slow? DO you have flaky tests?
		- ==Use agents to incrementally automate these things.==
		- The high leverage parts of our work is to:
			- ==Define the work that must be done==
			- ==Prioritize and schedule that work==
			- ==Effectively empower folks on our team to do that work==
		-  The more we can delegate and move into this sort of sequencer/orchestrator role... the more parallel and the more, deeper individual executions of those delegations to we're able to do.
		- If I put put primitives in place that make it super easy to spin up ways to respond to events on Kafka Queue, I don't need to be in the weeds with every engineer to make sure that they implement a consumer well.


Usually:
- Kick off task before I leave
- Tether my laptop to my phone
- Buckle it into the backseat
- Let it cook
-  Most of the time... with the skills we invoke that tell the agent to operate on a task until the tests are green, 


==Mistakes = codew ritten by an agent that is misaligned due to some not yet written down non-funcitonal requirement.==


____________

Juniors worried about their careers should be learning how to use agents effectively. the cheap open models are good enough for this now. it’s easier than ever to have a hireable portfolio. i think the people who are half-decent at getting LLMs to speed up development without creating a mountain of slop are gonna be fine for a long time. every company has a million unbuilt ideas that they’d love for someone to be responsible for. responsibility and local complexity awareness is a human bottleneck regardless of how good the agents are.

> Will, Interconnect discord



















