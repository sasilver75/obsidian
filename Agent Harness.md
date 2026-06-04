---
aliases:
  - Agent Scaffold
---


References:
- OAI Blog: [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/) ([[Ryan Lopopolo]], February 11, 2026)
- Video: [How Cursor Builds agentic Workflows across the SDLC](https://youtu.be/dJAVS1g3NDw) (May 12, 2026)
- Video: [Extreme Harness Engineering; Ryan Lopopolo on Latent Space](https://youtu.be/CeOXx-XTYek) (April 2026)


A program that wraps an LLM and turns it into an agent. The thing sitting between the model and the outside world.
- [[Claude Code]] itself is a harness, as is any homegrown loop your write around the API.

Concretely, a harness is responsible for:
- The ==agent loop==: 
	- Call the model, get a response, run any tool calls, feed results back, repeat until done.
- Tool execution: 
	- Actually running `Bash`, `Read`, `Edit`, MCP servers, etc. The model only emits a tool request; the harness decides whether and how to execute it.
- Permissions and safety:
	- Enforcing allow/deny rules (your `settings.json`), prompting you to approve a command, sandboxing.
- Context management:
	- Assembling the system prompt, injecting `CLAUDE.md`, compacting history when it gets long, handling cache.
- I/O with the user:
	- Rendering output, accepting input, slash commands, status lines, hooks.
- State outside the model:
	- Files on disk, todo lists, memory, sessions

The model is stateless and can't *do* anything on its own. ==The harness is what gives it hands.== 



_____________________

Video: [How Cursor Builds agentic Workflows across the SDLC](https://youtu.be/dJAVS1g3NDw) (May 12, 2026)


> "It's really cool to see the culture emerge; last night, we have a DevX channel where SWEs go to complain about their devx, and someone said that its' unfair that agents in the cloud don't have a place to complain, and now we have an AgentX channel where the agent express frustrations about not being able to do things, and we're seeing if we can autofix things that the agents can complain about."


_________

OAI Blog: [Harness Engineering: Leveraging Codex in an Agent-First World](https://openai.com/index/harness-engineering/) ([[Ryan Lopopolo]], February 11, 2026)
- Over the past five months, our team has been running an experiment: Building an shipping an internal beta of a software product with 0 lines of manually-written code: Every line, from application logic, tests, CI configuration, documentation, observability, and internal tooling.
	- We estimate we have built this in about 1/10th the time it would have taken to write the code by hand.

Five months later, the repository contains on the order of a million lines of code across application logic, infrastructure, tooling, documentation, and internal developer utilities. Over that period, roughly 1,500 pull requests have been opened and merged with a small team of just three engineers driving Codex. 

The lack of hands-on human coding introduced a different kind of engineering work, focused on ==systems==, ==scaffolding==, and ==leverage==.

==Early progress was slower than we expected,== not because Codex was incapable, but because the environment was underspecified. 
- ==The primary job of our engineering team became enabling the agents to do useful work.==

When something failed, the fix was almost never “try harder.” Because the only way to make progress was to get Codex to do the work, human engineers always stepped into the task and asked: ==“what capability is missing, and how do we make it both legible and enforceable for the agent?”==

Humans may review pull requests, but aren’t required to. Over time, we’ve pushed almost all review effort towards being handled agent-to-agent.

As code throughput increased, ==our bottleneck became human QA capacity==.

We’ve worked to add more capabilities to the agent by making things like the application UI, logs, and app metrics themselves directly legible to Codex.

For example, ==we made the app bootable per git worktree==, so Codex could launch and drive one instance per change.

We also wired the ==Chrome DevTools Protocol into the agent runtime and created skills for working with DOM snapshots, screenshots, and navigation==. This enabled Codex to reproduce bugs, validate fixes, and reason about UI behavior directly.

==We did the same for observability tooling. Logs, metrics, and traces are exposed to Codex via a local observability stack that’s ephemeral for any given worktree.==
- With this context available, prompts like “ensure service startup completes in under 800ms” or “no span in these four critical user journeys exceeds two seconds” become tractable.

![[Pasted image 20260512221438.png]]


![[Pasted image 20260512221527.png]]

We regularly see single Codex runs work on a single task for upwards of six hours (often while the humans are sleeping).


==Context management== is one of the biggest challenges in making agents effective at large and complex tasks. One of the earliest lessons we learned was simple: ==give Codex a map, not a 1,000-page instruction manual==.

We tried the “one big `AGENTS.md`” approach. It failed in predictable ways:
- ==Context is a scarce resource==. A giant instruction file crowds out the task, the code, and the relevant docs—so the agent either misses key constraints or starts optimizing for the wrong ones.
- ==Too much guidance becomes non-guidance==. When everything is “important,” nothing is. Agents end up pattern-matching locally instead of navigating intentionally.
- ==It rots instantly==. A monolithic manual turns into a graveyard of stale rules. Agents can’t tell what’s still true, humans stop maintaining it, and the file quietly becomes an attractive nuisance.
- ==It’s hard to verify==. A single blob doesn’t lend itself to mechanical checks (coverage, freshness, ownership, cross-links), so drift is inevitable.

==Instead of treating AGENTS.md as the encyclopedia, we treat is a table of contents==
- The repository’s knowledge base lives in a structured `docs/` directory treated as the system of record. A short AGENTS.md (roughly 100 lines) is injected into context and serves primarily as a map, with pointers to deeper sources of truth elsewhere.


```
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   ├── core-beliefs.md
│   └── ...
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   ├── new-user-onboarding.md
│   └── ...
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   ├── uv-llms.txt
│   └── ...
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```
Above: The in-repository knowledge store layout

Design documentation is catalogued and indexed, including verification status and a set of core beliefs that define agent-first operating principles.
- [Architecture documentation](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html) provides a top-level map of domains and package layering.

==Plans are treated as first class artifacts==.
- Ephemeral lightweight plans are used for small changes, while complex work is captured in execution plans with progress and decision logs that are ==checked into the repository==.
- Active plans, completed plans, and known technical debt are versioned and colocated ==so that agents can operate without relying on external context.==

This enables ==PROGRESSIVE DISCLOSURE:== Agents start with a small, stable entry point and are taught where to look next, rather than being overwhelmed with context up front.

==We enforce this mechanically==
- ==Dedicated linters and CI jobs== validate that the knowledge base is up to date, cross-linked, and structured correctly.
- ==A recurring "doc-gardening agent"== scans for stale or obsolete documentation that doesn't reflect the real code behavior, and opens fix-up PRs.

==Agent legibility is the goal==
- ==Our human engineers' goals is to make it possible for an agent to reason about the full business domain directly from the repository itself.==
	- From the agent's point of view, anything it can't access in-context while running effectively doesn't exist.
		- Anything in Google Docs, slack chat threads, or peoples' heads isn't accessible to the system; repository-local, versioned artifacts are all it can see.


![[Pasted image 20260512223553.png]]



Giving Codex more context means organizing and exposing the right information so that the agent can reason over it, rather than overwhelming it with ad-hoc instructions.
- ==Giving the agent the same information you would to a new teammate leads to better-aligned output (product principles, engineering norms, team culture)==

This framing clarified many tradeoffs; we favored dependencies and abstractions that could be fully internalized and reasoned about in-repo.

Technologies often described as "boring" tend to be easier for agents to model, due to composability, api stability, and representation in the training set.


_____________

[OpenAI Build Hour: Agents SDK](https://youtu.be/tK32trvj_b4)


Talking about the notion of a ==model-native harness==, since models are now being trained in their proprietary/in-house harnesses during post-training (model and harness are basically merging into one system). He talks about how it ships with tools, affordances, etc. that are "in-distribution" to the model, and how that might give better performance.
![[Pasted image 20260603185024.png]]
Left side is supposed to show how agent development looked in 2025, where you'd have to build all of:
- The Agent Loop: receiving requests from uesrs, routing ot model, calling tools, updating context, generating respones
- The Tool Integration Code: integration code, file search, web search, mcps, code interpreters, skills
- The Components: message handling, context management

And the idea is that [[OpenAI Agents SDK]] is supposed to solve many of these problems for you.
- "Instead of spending most of your time building the orchestration layer, you should spend more time building out your product to make it better for end users, and not really think about the orchestration."
- You get all of these tools out-of-the box, and hav added in some [[Sandbox]] functionality too.



If you imagine a world where the harness is tied to compute (codex running on laptop):
- You have a container where the agent is both running (loop calling LLM api, in addition to working on same filesystem is together)
- Your sandboxes become load-bearing, and if one dies or goes away, all that state is gone, and you don't have an external place you can refresh from. You have to do interesting gymnastics to manage your secrets; You don't want to have any secrets in our [[Sandbox]] or you'll be vulnerable to [[Prompt Injection]]/exfiltration.

If you split compute/harness, you can treat the sandbox as a totally ephemeral thing.. and the harness, which is maybe running in a [[Temporal]] job, or on AWS, which can handle the rehydration, snapshotting, all this sort of stuff.
![[Pasted image 20260603185408.png]]
Above:
- On the left, that harness runs inside the same sandbox where code execution happens. Teh sandbox also has the filesystem, and because the sandbox is untrusted, you cna't put serets or direct internal API access there, so you need a separate gateway service to hold secrets and intercept outbound calls.
	- If the sandbox is compromised/killed, your agent loop is affected too.
- On the right, the harness runs in a trusted service environment, outside the sandbox. The sandbox is reduce to what it *should* be: Just a place to run shell commands, inspect files, grep, build, test, and write code.
	- So the agent loop stays with the server, secrets, model calls, MCP/tool access, and durable orchestration, while the sandbox only receives bounded compute/file operations.
	- The sandbox has a more narrow job: execute commands and expose a filesystem. The harness has the broader job: reasoning, policy, tool routing, persistence, retries, approvals, and access to trusted services.


![[Pasted image 20260603190343.png]]
- Comes fully packed with ton of features like web search, file search, agent memory, text, and some other modalities (TTS, STT).
- Networked containers is a new thing in the API/SDK....
- We've also added agent memory so your tasks can improve over time and get better as they go.
- Autocompaction, computer use through the shell, async shell interaction loop similar to Codex...  
- Sandboxes are crucial: Modern work means working over many files (e.g. a large codebase, a lot of PDFs, creating powerpoints), and agents need access to those files and a place to create/store them. Sandboxes are an isolated environment where an agent has access to some files and can do stuff and produce meaningful output.
- The Agents SDK is open source an customizable; any model, even not OpenAI ones, that uses the OpenAI Responses format will work


![[Pasted image 20260603190808.png]]



![[Pasted image 20260603190813.png]]
More security control over containers


![[Pasted image 20260603190824.png]]
If you're used Skills in the past, common problem: You  a central source of truth for the skills, whether that's Github or a Bucket or now we have a Skills API where you can upload your skills.
-  A skill is a bundle of files, a SKILL.md file, but also scripts, etc that a model can use to do a specific task.
	- A task prep skill might define all the things I might need to know to do someone's taxes in 2025. 
	- With the Skills API, you can upload that, iterate on it over time and create versions, set a default version, reference those versions with a hosted shell, and all of that works pretty seamlessly.












