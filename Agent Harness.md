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
















