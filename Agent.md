References:
- Article: [Ampcode: How to Build an Agent](https://ampcode.com/notes/how-to-build-an-agent) (in 400 LoC)


An AI system that can take actions autonomously to accomplish a goal, rather than just generating a single response.

Key properties:
- ==Perception==: Observing its environment (==tools==, ==context==, ==memory==)
- ==Reasoning==: Planning steps toward a goal
- ==Action==: Executing tools (web search, code execution, file reads, API calls, etc.)
- ==Iteration==: Looping through observe -> reason -> act loops until the goal is complete.


![[Pasted image 20260603173446.png]]
Can be improved along all of these dimensions!



![[Pasted image 20260426195523.png]]
The evolution of the programmer, 2024-2026

![[Pasted image 20260502164011.png]]



___________________

All of our coordination tools (Github, Slack, JIRA, Linear) are from another era and not designed for the agentic development world; We funnel agentic outputs into platforms built for an outdated way of building software.

![[Pasted image 20260426210936.png]]

Dev process used to look like this, with plan/build/review phases, with times for touchpoints on the way, where everyone could give their 2 cents, get advice from across the team, catch mistakes, and course correct if things are going wrong.

By the time that the merge happened, everyone was on the same page!

![[Pasted image 20260426211015.png]]

The implementation window has collapsed.

Which means we think we don't need to plan as much, so most of our early touchpoints disappear
![[Pasted image 20260426211029.png]]


And review time for generated code is increased, but the alignment is on the "wrong" side of the implementation.
![[Pasted image 20260426211053.png]]

The time between logging an issue and an agent opening a PR is a couple of minutes. The code is so cheap that we don't really stop to think before we prompt it into existence.

![[Pasted image 20260426211127.png]]


Unhelpfully, most coding agents (e.g. CC) also have a local plan mode that is completely unshared with other people, so it's not checked. This leaves the weight of all the alignment to sit on the pull request, and it's never what PRs were really designed to do in the first place.

We're all experiencing the repercussions of this:
- ==Wasted work== (features no one asked for, don't solve problems, receiving feedback after you finished something)
- ==Coordination debt== (hairy merge conflicts from agents changing same files, developers duplicating work, etc.)

We need tools that help everyone on the team align BEFORE the agents statr working.
![[Pasted image 20260426211305.png]]
Planning and building are not separate phases, they are a cycle.
- We need to bring planning and development under one roof.


==Most of the context you need to build the right thing is not in the code base, it's in peoples' heads.==
- Business context and financial resources
- Political dynamics
- Product vision
- User research insights
- Organizational history

These matter immensely! Agents can't discover this context on their own; humans need able to share it early and anturally without adding process and overhead.

Github has built a new ressarch prototype: ACE (Agent Collaboration Environemnt)
- About to go into technicali preview.

![[Pasted image 20260426211440.png]]
Looks a bit like Slack, Github, Copilot had a baby
- Sessions are where you do work, they're multiplayer chats... but we also have coding agents in there.
	- They're backed by a micro VM, a sandboxed computer in the cloud on its own git branch.
		- The changes we make in each session are isolated; we can work on parallel tasks and switch between them.
- You jump into their sessions and see what they're doing, including their entire prompting history with the agent.



![[Pasted image 20260427130119.png]]


![[Pasted image 20260427130129.png]]

![[Pasted image 20260427130145.png]]

![[Pasted image 20260427130152.png]]


![[Pasted image 20260427130209.png]]

![[Pasted image 20260427130217.png]]


![[Pasted image 20260427130234.png]]


