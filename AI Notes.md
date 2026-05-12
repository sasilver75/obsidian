[Harness Engineering: How to Build Software When Humans Steer, Agents Execute — Ryan Lopopolo, OpenAI](https://youtu.be/am_oeAoUhew) (April 2026)


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



What does it mean to do a good job?
- It requires us years of being in the industry to fully internalize what it means to write high-quality, maintainable, reliable code that our teammates can build on top of, which will accrue leverage to the codebase.
- To do a single patch well requires 500 little decisions along the way re: the underspecified non-functional requirements.

It's our job to specify those non-functional requirements, and to write them down in a way that the agents can see. If the agent's aren't doing that, it's our job to refine and restrict their output such that the code they write is acceptable.
- Say: "Do not produce slop.", but to do that requires taking short-term velocity hits to back up or double-click into a task to figure out what the agent is struggling with in *your environment,* put the guardrails in place to help them, and then step back and try to solve higher leverage things.

We need to be continually refreshing context as the agent goes about doing a task. The ways we can do that are by having reviewer agents look at the code along the way, through the lens of what it means to be successful.
- We have ==security and reliability review agents== that are continually running as part of every push and CI that look and documentation and proposed patch and do simple things like ask: 
	- "Are there timeouts and retries on this network code?"
	- "Has the code that's been introduced have a secure interface that's impossible to misuse?"
- Everyone's been paged for network code that could have been remediated with a retry and timeout.
- Taking some time to write some docs... write some lint that look s at every time that I call `fetch`, I have a retry and a timeout wrapped around it. This means that I've durably solved this problem, and I'm able to do it because I lean on this axiom that code is free, and that agents do a good job, and that I can completely migrate the codebase to solve this problem durably once and for all.
- ==To operate in this way, we need to step back and look at the durable classes of failures that the agents and humans in the codebase are making time after time, figure out why we're spendign time on it, devicse a solution to systematically eliminate this class of behavior, and then continue to observe, refine, and make choices on these non-functional requirements.==
- You can write tests of source code that are separate from lints:
	- If we know that context is limited, we can write a test that limits files eto being no longer than 350 lines.
	- We can do a little bit of engineering to be context-efficient, and squeeze a litttle bit more juice out of the model capability that we have today.
- We can design error messages that give actual remediation steps to the model and to humans for how to perceive next. It's not enough to say that we've got a lint failure because we're awaiting in a loop, or because we have an unknown at a deep part of the codebase...
	- We provide a prompt via lint or test failure that says: "You shouldn't have an unknown here at all; we parse, don't validate at the edge"


You can just prompt things... 
- Prompts
- Powers are prompts
- Rules files are prompts
- Skills are prompts
- Lint error messages are prompts
- Review agents that inject comments into the PR that we require the agent to address before it's able to propose it for merge? prompts.

You can embed Agent SDKs into your tests that review your codebase for acceptability using prompts that get embedded into the code. If I find myself spending a bunch of time writing prompts, we can shell out agents to that as well.

Pointing Codex at the prompting cookbooks we have on the OpenAI developer guide and synthesizing a skill for how to write prompts, so that when I have a need to write a prompt to improve my agent performance locally in the code, I use the skill to write prompts that I wrote with the agent looking at the prompts to write the prompts!

All the leverage you encode into your repository, team, agents... stacks incredibly well.
A single product-minded engineer was able to give a good lift.
- To write a good QA plan, you need to be able to document all the features you have, critical user journeys, how users engage with your applications, web apps, apis, and services.
- Once you write these down... on how to write a good QA plan, with the expectation that all user-facing work HAS a QA plan, an review agent can assert expectations around what it means to prove that you've effectively written a feature. A QA plan indicates what media should be attached to a PR for humans and agents to know what should be attached to a PR for the humans and agents to know that you've done a good job, which has a consequence of me trusting the output more, shoulder surfing less, and removing myself from the loop more to delegate more work to agents.



1. Start wth tickets: Features we wan to add, chunks of rwork, reliability we wnat to do
2. Give that work to agents with a couple o fskills to manipupate the app
3. We want the entry point to the devlopment process to be codedx, not an environment qwe builda round it, so we kind ofdo things outside it; codex is the entrypoint, and we give it instructions on how to cook. Rather than giving it a shell that our app and codex gets pawned into, we have a skill that teaches codex how to launch the app, how to spin up the localo bservability stack to geive it logging and telemetry, or how to boot up chrome dev tools and attach to the application with a local cli that we connect via some daemon that we have. 

The wholw ay we've set up teh repo and all local dev tools is for codex to invoke them first

So we have  bunch of litttle mini-arnesses in teh code base that make it easy for us to slot in guardrails

For example a big package of custom ESLint rules that get wired in to every pnpm package in teh workspace.

Another local dev harness that lets us add higher-level whoelsome tests that let us assert the stsructure of the code itself, rather than teh syntax or behavior of the code; things like package privacy, dependency edges betwen different layers of the stack, etc. Making sure that across multiple files, Zod schemas are deduplciated, that there's a single canonical implementation of oru async helpers, these sortso f things.
- Because The way we've SEEN these agents owrk is to optimize  for local coherence of package rather than using our shared utilities, and things like that
- Having observed this behavior, ew've built a numbero f pseudo linter source code verification things that shake out some of that bad behavior so that humans don't get distracted paying attention to that in reviews.
- The setup optimizes for the agent to do the job, and for humans to not have to keep track of the high churn in the codebase.

We've centralized our leverage on 5-10 skills, we don't go super wide with skills, preferring to make the existing skills better.
- I find that... the infrastructure in the repo, all the local developer tools, change super frequently, and I don't have the bandwidth to keep track of this, and so we hide all of this complexity below the skills that the human has to invoke... and let the agent just kind of figure it out.
    - When we moved to chrome dev tools directly to having this daemon thing... I didn't know that had happened for 3 weeks!


Q: How do you stop yourself from over-engineering harnesses? Do you often build small, custom tools for yourself?

A: This is gesturing in the direction of the bitter lesson: How do I make sure that the work that I do isn't completely obsoleted by an increase in model capability.
A: the way I thin about this is doing the bare minimum amount of context management to pull in requirements for the agent to do an acceptable job over the course of its work. Context is a thing that won't ever be obsoleted: Understanding requiresments of the task, which guardrails to pay attention to, etc.
==A: A good harness is about giving the model text at the right time so that it can look at the work it's done, and hte informaiton about what a good job looks like.==

All the harness should do is surface instructions to the model at the right time.

==figuring out ways to defer or JIT surface those instructions is what a good harness should do==. If you know that you want your React components decomposed so that they make good snapshot test for individual, more stateless pieces, you don't need to load that up front, you should let the gent cook/code/experiment with the UI, and then at lint or test time, say: "Okay, you've done the work. To finish this, you should break this apart so that the components are as small and stateless as possible, and have local dependencies on hooks instead of prompt driving, or whatever it is you want your code to look like."


Q: A lot of people talk about the Codex harness... how does it compare to Claude code, etc... 
A: 
- ==One thing that is super powerful is this ontion that the labs are not just posttraining hte models, but also posttraining them in the context of teh models in which they are primarily deployed in.==
	- The "ApplyPatch" tool, or the specific quoting semantics of how to invoke the bash tool... are IN THE LOOP for the posttraining process for the hanresses from the labs, so there's leverage to be had by depending on tehse first-party harnesses directly.
	- Being able to direct through them... menas you get to ride the wave of the post-training ... and focus on code.
- I have confidence that CC and Cdoex will continue to get better.



______
[Extreme Harness Engineering: 1M LOC, 1B toks/day, 0% human code or review — Ryan Lopopolo, OpenAI](https://youtu.be/CeOXx-XTYek)



In the prompts we give these agents, we do allow them to push bakc. When we first added coding agents... to PRs, then the review agent would fire on PR, and we post a comment, and instruct codex to acknowledge and respond to that feedback... and initially, the Codex driving the ocde author as willing to be bullied by the PR reviewer, so you could end up with a thing where ==the PR wasn't converging!==

So:
- Review agent were then biased towards merging the thing, and to not surface things greater than P2 in priority (we didn't DEFINE P2, but gave it a framework to think about...)
- On the code authoring agent side, we gave it the flexibility to defer or push back against agent feedback.
	- I happen to notice something in the code review which could blow up scope by a factor of 2x?... then that doesn't need to be addressed, it's more of an FYI, file it as a backlog ticket, pick it up in the next "fix things" week.

Otherwise agents bias to what they were trained to do, which is follow instructions.


==So the coding agent can merge autonomously.== This is something that people are uncomfortable with, typically.


Q: A lot of how teams are structured, write code, do review, is for human legibility and operability of software. A lot of what you're saying is pretty drastic; should people full-send?
A:
- I'm very removed from the process; I can't really have deep code-level opinions about things, it's as if I'm group tech leading a 500 person organization; it's not appropriate for me to be in the weeds for every PR.
- I have some representative sample of the code as it's written, and I have to use that to infer:
	- What the teams are struggling with
	- Where they need help
	- Where they are moving quickly so that I can spend time elsewhere
- I do have a Command base class which is used to have repeatable chunks of business logic that comes with tracing, metrics ,and observability for free
	- The thing to focus on is not how business logic is structured, but that it uses this primitive, cuz I know that this gives leverage by default.


Q: Other thing I want to pull on. You mentioned ticketing systems and PRs; do both of these have to be reinvented for this new kind of coding? Git itself is somewhat hostile to multi-agents.
A: 
- WE make very heavy use of worktrees
Q: Even then, we did a podcast with Cursor and hey said they're getting rid of worktrees because there are too many work conflicts, etc.
A: The models are erally great at resolving merge conflicts;
- To get to a state where I'm just not synchronously in the loop in my terminal, I almost don't care that there are merge conflicts.
- We invoke a $LAND skill, which coaches codex to push the PR, wait for human and agent reviewers, wait for CI to be green, fix the flakes if there are any, merge the upstream if the PR comes into conflict, wait for everything to pass, put it in the merge queue, deal with flakes until it's in main.
	- This is what it means to delegate fully.
	- In a large monorepo, it's a huge tax on humans to get this merged, and agents are capable of doing this very well.


Ultimately, all the things w'eve encoded in docs and test are thew ays of putting the nonfunctional requirements of building high-scale reliable, high-quality software into a place where we can inject it as prompts the agent
- Docs
- Lints
The whole meta of the thing is to tease out of the heads... from all the engineers on their team.. what good looks like, what they would do by default, and what they would teach new hires
- This is why it's very important to look at the mistakes that agents make; 
	- This is code being written that's misalligned with some as-yet-not-written down functional requirement


A: Folks were taking the link to the article, giving it to Pi or Codex, and saying "Make my repo this", and it was WILDLY EFFECTIVE.
Q: I tried with 5.4 yesterday... can we just scaffold out what it would be to run this? It was a good way to learn what could be changed!


A: re: Specs... talking about Symphony in a little bit. The way we distribute it as a spec... people are callign tehse [[Ghost Library|Ghost Libraries]], which is a cool name.
- It means that it becomes cheaper to share software with the owrld
	- Define spec
	- How you could build ryour own
	- Specifying as much as is required for your coding agent to build it locally
- We have taken all the scaffolding in our proprietary repo, spun up a new one, asked Codex with our repo as a reference, write the spec, told it to spawn a disconnected codex to implement the spec, wait for it to be done, spawn another Codex in another tmux to review the implementation compared to upstream, and update the spec so that it diverges less, and then just loop ralph style until you have a specification that is able to, with high fidelity, able to reproduce the system as it is.
	- Q: And you're not really adding any of your human bias in there; sometimes people write a spec and think "I think it should be done this way, etc." The agent can determine the spec better. 
	- Q:Can an agent produce a spec it can't solve?
		- ==A: With symphony, there's an axis... you have things that are easy or hard, established or new. I think things that are hard and new.. it's still something that the models need humans to drive. but the other quadrants are likely "solved"==
		- This means the humans with limited time and attention get to work on the hardest stuff, or the deepest refactorings where you don't know what the proper shape of the interfaces are.

Okay, let's introduce Symphony! See [[OpenAI Symphony]] for notes

Q: Okay, and I see you use Linear... 
A: 
- We do make good use of Slack too, we fire off Codex to do all these low complexity fix-ups... things that sync that knowledge into the repo... it's super cheap.
Q: My biggest plug is that OpenAI needs to rebuild Slack.
A: I've brought this up. If we think we want agents to do economically valuable work...then we need to find ways for them to naturally collaborate  with humans, which means that collaboration tooling (github, slack, linear) is a naturally interesting thing to explore.

Q: yeah, there's no great team collaboration for Doex; it seems your team has some say into what comes out... if you guys are on the bound... what's stuff that you might not focus on... but what do you expect other people to be building? 
- Should we build stuff that's... very niche for our workflow, for our team? Should it be more general so that other people acn adopt it?
- Is everything just internal tooling now? the way that our team operates and likes to communicate...
	- A: TBD, I think. Ther's leverage to be had in making the code and processes as much the same ans possible. IF you think that coei s context, code is prompts, then it's better for an agent behavior perspective for it to be able to look in a package in directy XYZ and not have ot page so deply into ABC because they have sam structure, the same language, the same patterns internally; the this comes from aligning on a single set of skills that you're pouring every engineers skills and tastes into to make sure that the agent is effective.
	- In our codebase we have six skills. if some part of the software development skill is not covered, the first attempt is to encode it in an existing setup skills.
		- So we can change agent behavior more cheaply than changing the human driver behavior.

Q: Have you experimented with agents changing their own behavior, or agents changing their own subagents behavior?
A: We have some experiments... we can point codex at its own session logs and ask it to how to use its own tools better.
	How can I use this skill session better?
	What skills should I have?
A: Yeah, You can just codex things. Just prompt things. It's really glorious.

==A: We're actually slurping these session logs into object storage and running agents over them evrey day to figure out how we can, as a team, do better, and reflect that back into the repository so that everyone can beefit from everyone else's behavior for free.  Same for PR comments: these are all feedback about code deviating from what was good; a PR comment, a failed build, these are all signals that mean that at some point the agent was missing context, we have to figure out how to slurp it up and put it back in the repo.== 

Q: When I use Claude Cowork for knowledge work... I always have ask it what I should do better next time...

Q: So you have 6 abstraction levels in Symphony:
- Policy
- Configuration
- Coordination
- Execution
- Integration
- Observability

But the 0-th layer is: "Okay, are we working well, can we improve how ew work? Can I modify my own workflow.MD?"
A: yeah, this thing is also able to cut its own tickets, because we give it full access. We're able to put in the ticket that you expect it to follow up on. Don't put the agent in the box, give the agent full accessibility over its domain.


Q: Okay, a lot of software, has a gui, and it's made for humans. We're seeing a lot of CLI evolution; do we get good vision, do we get good sandboxes? Right now, models love to use tools, read through text, slap a CLI and let it go use
A: Yep! WE've also been adapting non-textual things to that shape to improve model behavior on some ways. WE want the agent to be able to see the UI, but agents don't perceive visually in the same way that we do. They see things in latent space... If we want it see the actual layout, it's almost easier to rasterize that image to ascii art and feed it in to the agent. There's no reason that you can't do both to further refine how the model perceives the object it's manipulating.

Q: Talk more on these layers in Symphony?
A: The Coordination Layer is a tricky piece to get right. This is where we turn the spec into Elixir...and the model takes a shortcut; oh, I have all these primitives I can use in this runtime that has native process subervision! It's a neat way to take the spec and make it more achievable by making choices that naturally map to the domain. In the same way that you'd prefer to have  TS monorepo for full stack web development (ability to share types reduces complexity).
Q: This is what GraphQL used to be, idk if it's still alive
A: ==My personal ability to write or not write elixir doesn't have to bias us away from using the right tool for the job, which is wild==

Q: This is just an interesting hierarchy to travel up and down and to think about.
A: The Policy layer is super interesting! You don't have to build a bunch of code to make sure that the system waits for CI to pass.
Q: It's your institutional knowledge
A: Yeah, just give it the GH CLI with some text saying that the CI has to pass! Makes maintenance easier

A: In our pnpm distributed script runner, when you do --recursive, it produces a mountain of text, but all of that is for passing test suites, so we ended up wrapping all of this in another script (vibe coded) to only output the failing parts of the tests.

The agents are good at following instructions, so give them instructions. It wil improve the reliability.
We don't want to have people monitor the agent and they are vibing it into existing.
Being very opinionated and strict about the success criteria area mans that our deployment success rate goes up, which means we dont' have to get tickets...

Q: This goes back to "code is disposable". Earlier you'd kick off a 2h codex run, you'd want to monitor... I don't want it to go down the wrong path, cut if off, etc. But no, just 4X it in parallel! One might be right, one might be better, don't overthink it.
A: Yeah




