Multi-[[Agent]] orchestration system for [[Claude Code]] and others, with persistent work tracking.
- Lets you coordinate multiple coding agents working on different tasks.
- Instead of losing context when agents restart, Gas Town persists work state in git-backed hooks, enabling reliable, multi-agent workflows.

![[Pasted image 20260426181407.png]]


Architecture:![[Pasted image 20260426181439.png]]
- ==The Mayor==: Primary AI coordinate; a Claude Code instance with full context about your workspace, projects, and and agents. 
	- You tell the mayor what you want to accomplish!
- ==Town==: Your workspace directory (e.g. ~/gt/), which contains all projects, agents, and configuration.
- ==Rigs==: Project containers. Each rig wraps a git repository and manages its associated agents.
- ==Crew Members==: Your personal workspace in a rig where you do hands on work.
- ==Polecats==: Worker agents with persistent identity but ephemeral sessions. Spawned for tasks, sessions end on  completion, but identity and work history persist.
- ==Hooks==: Git worktree-based persistent storage for agent work. Survives crashes and restarts.
- ==Convoys==: Work tracking units; bundle multiple beads that get assigned to agents. Convoys labeled `mountain` get autonomous stall detection and smart skip logic for epic-scale execution.
- [[Beads]] integration: Git-backed issue tracking system that stores work state as structured data, using prefix+5-char formats like `gt-abc12`, `hq-x7k2m`. 
	- Commands like `gt sling` and `gt convoy` accept these IDs to reference specific work items.
	- Terms "bead" and "issue" are used interchangeably; beads are the underlying data format, while issues are the work items stored as beads.
- ==Molecules==: Workflow templates that coordinate multi-step work. Formulas (TOML definitions) are instantiated as molecules with tracked steps. Two modes:
	- Root-only wisps (steps materialized at runtime, lightweight)
	- Poured wisps (steps materialize as sub-wisps with checkpoint recovery)

==Monitoring==: A three-tier watchdog system keeps agents healthy:
- ==Witness==: Per-rig lifecycle manager. Monitors polecats, detects stuck agents, triggers recovery, manages session cleanup.
- ==Deacon==: Background supervisor running continuous patrol cycles across all rigs.
- ==Dogs==: Infrastructure workers dispatched by the Deacon for maintenance tasks (e.g. Boot for triage)

==Refinery==: Per-rig merge queue processor. When polecats complete work via `gt done`, the Refinery batches merge requests, runs verification gates, and merges to main using a Bors-style bisecting queue. Failed MRs are isolated and either fixed inline or re-dispatched.

==Escalation==: Severity-routed issue escalation; Agents that hit blockers escalate via `gt escalate`, which creates tracked beads routed through the Deacon, Mayor, and (if needed) ==Overseer== (you). Security levels: CRITICAL (P0), HIGH(P1), MEDIUM (P2).

==Scheduler==: Config-driven capacity governor for polecat dispatch. Prevents API rate limit exhaustion by batching dispatch under configurable concurrency limits.

==Seance==: Session discovery and continuation. Discovers previous agent sessions via `.events.jsonl` logs, enabling agents to query their predecessors for context and understand decisions from earlier work.
```
gt seance                       # List discoverable predecessor sessions
gt seance --talk <id> -p "What did you find?"  # One-shot question
```

==Wasteland==: Federated work coordination network linking Gas Towns through DoltHub. Rigs post wanted items, claim work from other towns, submit completion evidence, and earn portable reputation via multi-dimensional stamps.
- (Seems like this is some bullshit?)


![[Pasted image 20260426183256.png]]


# Common Workflows
![[Pasted image 20260426183343.png]]



# Activity Feed
- `gt feed` launches an interactive terminal dashboard for monitoring all agent activity in real time. It combines beads activity, agent events, and merge queue updates into a three-pane TUI:
	- Agent Tree: Hierarchical view of all agents grouped by rig and role
	- Convoy Panel: In-progress and recently-landed convoys
	- Event Stream: Chronological feed of creates, completes, slings, nudges, and more
- Still, at scale (20-50+ agents), spotting stuck agents in the activity stream becomes difficult. The ==problems view== surfaces agents needing human intervention by analyzing structured beads data.
	- You can press `p` in your `gt feed` (or run  `gt feed --problems`) to toggle this view, which groups agents by health state.
	- You can then press `n` to nudge the selected agent or `h` to handoff.

# Dashboard
- Gas Town includes a web dashboard for monitoring your workspace. The dashboard must be run from inside a Gas Town workspace (HQ) directory: `gt dashboard`
- The dashboard gives you a single-page overview of everything happening in your workspace: agents, convoys, hooks, queues, issues, escalations. 


# Monitoring and Health:
Gas town uses a three-tier watchdog chain to keep agents healthy at scale:
```
Daemon (Go process) ← heartbeat every 3 min
    └── Boot (AI agent) ← intelligent triage
        └── Deacon (AI agent) ← continuous patrol
            └── Witnesses & Refineries ← per-rig agents
```
- ==Witness== (Per-Rig): Each rig has a Witness that monitors its polecats. The Witness detects stuck agents, triggers recovery (nudge of handoff), manages session cleanup, and tracks completion. Witnesses delegate work rather than implementing it directly.
- ==Deacon== (Cross-Rig): The Deacon runs continuous patrol cycles across all rigs, checking agent health, dispatching ==Dogs== for maintenance tasks, and escalating issues that individual Witnesses can't resolve.

# Merge Queue (==Refinery==)
The Refinery processes completed ==Polecat== work through a bisecting merge queue:
1. Polecat runs `gt done` -> Branch pushed, MR bead created
2. Refinery batches pending MRs
3. Runs verification gates on the merged stack
4. If green: all MRs in batch merge to main
5. If red: bisects to isolate the failing MR, merges the good ones

This is a "Bors-style" merge queue; the ==Polecats== themselves never push directly to main.


# The ==Scheduler== controls ==Polecat== dispatch capacity ot prevent API rate limit exhaustion
```
gt config set scheduler.max_polecats 5   # Enable deferred dispatch (max 5 concurrent)
gt scheduler status                      # Show scheduler state
gt scheduler pause                       # Pause dispatch
gt scheduler resume                      # Resume dispatch
```
Default mode (`max_polecats = -1)` dispatches immediately via `gt sling`. When a limit is set, the daemon dispatches incrementally, respecting capacity.


# ==Seance==: Discover and query previous agent sessions
```
gt seance                              # List discoverable predecessor sessions
gt seance --talk <id>                  # Full context conversation with predecessor
gt seance --talk <id> -p "Question?"   # One-shot question to predecessor
```
Seance discovers sessions via `.events.jsonl` logs, enabling agents to recover context and decisions from earlier work without re-reading entire codebases.


# Telemetry
- Gas Town emits all agent operations as structured logs and metrics to any [[OpenTelemetry Protocol]] (OLTP) compatible backend (VictoriaMetrics/VictoriaLogs by default):
```
# Configure OTLP endpoints
export GT_OTEL_LOGS_URL="http://localhost:9428/insert/jsonline"
export GT_OTEL_METRICS_URL="http://localhost:8428/api/v1/write"
```
Events emitted: session lifecycle, agent state changes, bd calls with duration, mail operations, sling/nudge/done workflows, polecat spawn/remove, formula instantiation, convoy creation, daemon restarts, and more.


# Propulsion Principle
- Gas Town uses [[git]] hooks as a propulsion mechanism.
- Each hook is a git worktree with:
	- Persistent state: Work survives agent restarts
	- Version control: All changes tracked in git
	- Rollback capability: Revert to any previous state
	- Multi-agent coordination: Shared through git
![[Pasted image 20260426185505.png]]




![[Pasted image 20260426185517.png]]

![[Pasted image 20260426185524.png]]

![[Pasted image 20260426185531.png]]


___________________________________

[Steve Yegge Medium Post: Welcome to Gas Town (Jan 1, 2026)](https://steve-yegge.medium.com/welcome-to-gas-town-4f25ee16dd04)

- Gas Town helps you with the tedium of running lots of Claude Code instances. Stuff gets lost, it's hard to track who's doing what, etc. 
	- Gas Town helps with all that yak shaving, and lets you focus on what your Claude Codes are working on.
	- You can use 20-30 at once, productively, on a sustained basis.
- ==Gas Town is complicated.== Not because he wanted it to be, but because he had to keep adding components until it was a self-sustaining machine. 
	- Gas Town solves the MAKER problem (20-disc Hanoi towers) trivially with a million-step wisp that you can generate from a formula. The 20-disc wisp takes about 30 hours.
- In August (2025) he started building his own orchestrator, since nobody seemed to care... v2 was [[Beads]], and v3 was [[Gas Town]] (in Python), with v4 being Gas Town (in Go). 

![[Pasted image 20260426195505.png]]
The Evolution of the Programmer in 2024-2026, pictured by Nano Banana

Working effectively in Gas Town requires ==committing to vibe coding.==
- Work becomes fluid, an uncountable substance that you sling around freely, like slopping shiny fish into wooden barrels at the docks.
	- Most work gets done, some work gets lost, fish fall out of the barrel, some escape to sea, or get stepped on. More fish will come.
	- ==The focus is throughput: Creation and correction at the speed of thought.==

==Work in Gas Town can be chaotic and sloppy, which is how it got its name==
- Some bugs get fixed 2-3 times, and someone has to pick the winner.
- Other fixes get lost.
- Designs go missing and need to be redone.
The idea is that it doesn't matter, because you're churning forward relentlessly on huge, huge piles of work, which Gas Town is both generating and consuming.

In Gas Town, you let Claude Code do its thing. You are a PM, and Gas Town is an Idea Compiler. You just make up features, design them, file the implementation plans, and then sling the work to your polecats and crew.

==You have to keep Gas Town running.==
- It runs itself pretty well most of the time, but stuff goes wrong often.
- It can take a lot of elbow grease from you and the workers to keep it running smoothly.
- It's very much a hands on the wheel orchestration system.

==It's also expensive as hell.==
- You won't like Gas Town if you ever have to think, even for a moment, about where money comes form.
- You typically need ==multiple Claude Code accounts==, etc. It's a cash guzzler.

Gas Town uses `tmux` as its primary UI.

Like it or not, Gas Town is built on [[Beads]].
- It is in fact the SEQUEL to Beads.
- Beads is the Universal Git-Backed Data Plane (and control plane, it turns out) for everything that happens in Gas Town. You have to use Beads to use Gas Town.


This year, he expects agents to become more Gas Town-friendly, and Gas Town and Beads make it into the training corpus for frontier models. Even without all that, it's already shocking that the agents use Beads and Gas Town so efficiently with zero training.

![[Pasted image 20260426200512.png]]
Gas town workers are regular coding agents, each prompted to play ==one of seven well-defined worker roles.== 
- There are some other key concepts that I'll briefly introduce, along with the roles, like ==Towns== and ==Rigs==.

One thing to know up front: ==It degrades gracefully.==
- Every worker can do their job independently, or in little groups, and at any time you can choose which parts of Gas Town you want running.
- It even works in  `"no-tmux"` mode, and limps along using naked CC sessions without real-time messages. It's a little slower, but it still works.


# Key Players and Concepts

### The ==Town==
- This is your HQ, his is ~/gt, and all of his project rigs go beneath it: gastown, beads, wyvern, efrit, etc.
- The town (Go binary `gt`) ==manages and orchestrates all the works across all your rigs.==

### ==Rigs==
- Each ==project== (git repo) that you put under Gas Town management is called a Rig.
	- Some roles (==Witness==, ==Polecats==, ==Refinery==, ==Crew==) are per-rig, while others (==Mayor==, ==Deacon==, ==Dogs==) are town-level roles.
		- Again: ==SOME ROLES ARE PER-RIG, WHLIE OTHERS ARE PER-TOWN!==
- `gt rig add` and related commands mange your rig within the Gas Town harness. Rigs are easy to add and remove.

### The ==Overseer==
- ==You==, the human! You have an identity in the system, and your own inbox, and can send and receive town mail.


### The ==Mayor==
- The main ==agent you talk== to most of the time; your concierge and chief-of-staff.
- The Mayor ==typically kicks off most of your work convoys, and receives notifications== when they finish.

### ==Polecats==
- Polecats are ephemeral per-rig workers that spin up on demand.
- Work, often in swarms, to produce ==Merge Requests (MRs)==, then hand them off to the ==Merge Queue (MQ)==.
- ==After the merge, they're fully decommissioned==, though their names are recycled.

### ==Refinery==
- As soon as you start swarming workers, you run into the ==Merge Queue== (MQ) problem: Your workers get into a monkey knife fight over rebasing/merging, and it can get ugly.
	- Final workers to get their merge in can be trying to merge against an ==unrecognizable new head!==
	- They may need to completely reimagine their changes and reimplement them.
- This is the job of the ==Refinery:== the engineer agent ==responsible for intelligently merging all changes, one at a time, to main.==
	- No work can be lost, although it is allowed to ==escalate==.

### ==The Witness==
- Once you spin up enough polecats, you realize you need an ==agent just to watch over Polecats and help them get un-stuck==.
- Gas Town's propulsion (GUPP) is effective, but still a bit flaky now, and ==sometimes you need to hustle the polecats ot get their MRs submitted, and then hustle the Refinery to deal with them.==
- The Witness patrol helps smooth this out, and it's almost perfect for most runs.

### The Deacon
- The deacon is the daemon beacon, named for a character from Waterworld, inspired by a character from the Mad Max universe.
- The Deacon is a ==Patrol Agent==: It runs a "patrol" (a well-defined workflow) in a loop. Gas Station has a daemon that pings the Deacon every couple minutes and says: "Do your job." The Deacon intelligently propagates this DYFJ signal downward to the other workers, ensuring Gas Town stays working.

### Dogs (Town-Level)
- Inspired by Mick Herron's MI5 "Dogs" (from Slow Horses show), this is the Deacon's personal crew.
- Unlike Polecats, Dogs are town-level workers, and do things like ==maintenance (cleaning up stale branches, etc.) and occasional handyman work for the Deacon, such as running plugins.==
	- "The Deacon's patrol got so overloaded with responsibilities that it needs helpers, so I added the Dogs"
	- This helps keep the Deacon focused on completing its patrol, rather than getting bogged down and stuck on one of the steps.
	- ==the Deacon slings work to the Dogs, and they handle the grungy details.==

### Boot the Dog
- There is a ==special Dog== named Boot who is awakened every 5 minutes by the daemon, just to check on the Deacon.
- ==This is its only job.== Boot exists because the daemon kept interrupting the Deacon with annoying heartbeats and pep talks, so now the dog gets to hear it.
	- Boot decides if the Deacon needs a heartbeat, a nudge, a restart, or simply to be left alone, then goes back to sleep.

### The Crew
- ==The agents you'll personally use the most, after the Mayor.==
- ==per-Rig coding agents who work for the Overseer (you) and are not managed by the Witness.==
	- You choose their names, and they have long-lived identities; spin up as many as you like. The tmux bindings let you cycle through the crew in a loop for each 

==The Crew are the direct replacements for whatever workflow you used to be using.== It's a bunch of named CC instances that can get mail and sling work around.


### Mail and Messaging
- [[Beads]] are the atomic unit of work in Gas Town
	- A bead is a special kind of issue-tracker tissue, with an ID, description, status, assignee, and so on.
	- Stored in [[JSON Lines|JSONL]] and tracked in Git along with your project's repo.
	- ==Town mail and messaging (events) use Beads, as do other types of orchestration.==
- Gas town has a TWO-LEVEL Beads structure: 
	- Rig beads: Rig-level work is project work (features, bug fixes, etc). This work is split between polecats and crew, with other workers stepping in occasionally.
	- Town beads: Town-level work is orchestration (patrols: long strings of steps to follow, encoded as linked beads) and on-shot workflows like releases, or generating cross-rig code review waves.
- All rig-level workers (refinery, witness, polecats, and crew) are perfectly able to work cross-rig when they need to. There is a `gt worktree` command that they can use to grab their own clone of any rig and make a fix, but normally they work inside a single project/rig.
- Gas Town configures ==Beads== to route requests like `bd create` and `bd show` to ==route to the right database based on the issue prefix, like "bd-" or "wy-".== 
	- All bead commands work pretty much anywhere is Gas Town and figure out the right place to put them, and if not, its' easy to move Beads around.

![[Pasted image 20260426203432.png|1945]]



## Gastown Universal Propulsion Principle (GUPP)
- The biggest problem with CC is that it ends: The context window fills up, it runs out of steam, and stops. GUPP is Yegge's solution to this problem, and states:
	- ==If there is work on your hook, YOU MUST RUN IT.==
- All Gas Town workers, in all roles, have persistent identities in Beads (in git). A worker's identity type is represented by a ==Role Bead==, which is like a domain table describing the role. Each worker has an ==Agent Bead==, which is the agent's persistent identity.
- Both Role Beads and Agent Beads are examples of =="pinned beads",== meaning they float like yellow sticky notes in the Beads data plane and ==never get closed like regular issues==. They don't show up in `bd ready` (which shows ready work) and are treated specially.


Claude Code sessions are the cattle that Gas Town throws at persistent work.
That work all lives in Beads, alongside the persistent identities of the workers, the mail, the event systems, and even the ephemeral orchestration.

In Gas Town ==an agent IS a Bead,== an identity with a singleton global address.
- It has some slots, including a pointer to its ==Role Bead== (Which has priming information, etc. for that role), its ==mail inbox== (all Beads), its ==Hook== (also a Bead, used for GUPP), and some ==administrative stuff like orchestration state== (labels and notes). The history of everything that agent has done is captured in Git, and in Beads.

![[Pasted image 20260426214813.png|957]]

## Hooks
What is a ==Hook?==
- Every worker in Gas Town have their own hook, a special ==pinned bead just for that agent,== which is where you hang ==molecules,== which are Gas Town workflows.
- How does stuff get hung there? With `gt sling`! 
	- ==You sling work to workers, and it goes on their hook.==
	- You can start them immediately, or defer it, or even make them restart first.

One of the simplest but greatest things about Gas Town is that any time in the session, you can say: "let's hand off," and the worker will gracefully clean up and restart itself. Thanks to GUPP, the agent will continue working automatically if it's hooked.
- Claude is so miserably polite that GUPP doesn't always work in practice!
- We tell the agent ==YOU MUST RUN YOUR HOOK,== and it somethings doesn't do anything at all, just sitting there waiting for user input.

==GUPP Nudge== is the workaround "hack"
- Gas Town Workers are prompted to follow "physics over politeness," and are told to look at their hook on startup.
- If their hook has work, the must start working on it without waiting.
- Unfortunately, Claude often waits till you type something before it checks its mail and hook, reports in, and starts working; sometimes it does, sometimes it doesn't. 
	- ==There are various systems in place that will nudge the agent, roughly 30 to 60 seconds after it starts up.==
	- Sometimes faster, sometimes slower. ==But it will always get the nudge within 5 minutes or so==, if the town is running.

It doesn’t matter what you tell the agent in the nudge. Because their prompting is so strict about GUPP and the theory of operation of Gas Town, and how important they are as gears in the machine, blah blah blah, that agents will completely ignore whatever you type unless you are directly overriding their hook instructions.


## Talking to your dead ancestors
- `gt seance` ==lets GT workers communicate directly with their predecessors in their role.==
	- They do this with the help of CC's `/resume` feature, which lets you restart old sessions that you killed.
- A worker will often say: "Ok, I handed off this big pile of work and advice to my successor, bye!" and then the new worker spins up and says "I don't see shit!" Resolving this manually is awkward and not worth it.
- With gt seance, the worker will literally spin Claude Code up in a subprocess, use /resume to revive its predecessor, and ask it, “Where the hell is my stuff you left for me?”

## MEOW Stack (Molecular Expression of Work (MEOW))
- Gas Town may not live longer than 12 months, but the bones of Gas Town, the MEOW stack, ay live on for several years to come, feeling more like a discovery than an invention.

![[Pasted image 20260426215745.png]]
- ==A desire to break agent work up into sequenced small tasks that they must check off, like a TODO list.==, which they would execute atomically in the right order.
	- ==Molecules are workflows, chained with Beads.== They can have arbitrary shapes, unlike epics, and they can be stitched together at runtime.

==Protomolecules==: ==Classes or templates==, made of actual Beads, with all the instructions and dependencies set up in advance... an entire graph of template issues (e.g. "design", "plan", "implement", "review", "test" in a simple one), which you would then ==instantiate into a molecule.==
- The instantiation involves copying all the protomolecule beads and performing variable substitutions on it to create a real workflow.

Example:
- I have a 20 step release process for Beads.
- Agents used to struggle to get through it, because it had long wait states, like waiting for [[GitHub Actions]] to complete, for [[Continuous Integration|CI]] to finish, and for various artifacts to be deployed.... they'd often skip steps.
- With molecules, the idea was: Make 20 beads for  the release steps, chain them together in the right order, and make the agent walk the chain, one issue at a time. One added benefit is that it produces an activity feed automatically, as they claim and close issues.
- If the workflow is captured as a molecule, then it survives agent crashes, compactions, restarts, and interruptions.


We soon found out that we needed a macro-expansion phase to properly compose molecules with loops and gates, so we came up with a source form for workflows, ==Formulas== in [[Tom's Obvious Minimal Language|TOML]] format, which are "cooked" into protomolecules and then instantiated itno wisps or mols in the Beads database.

Formulas provide a way for you to describe and compose pretty much all knowledge work.


![[Pasted image 20260426220747.png|957]]

The term for the big seas of work molecules, all the work in the world, is "guzzoline," though we don't use it in the docs much.


## Nondeterministic Idempotence
- NDI, or Nondeterministc Idepotence, is similar to Temporal's deterministic, durable replay, but Gas Town achieves its durability and guaranteed execution through completely difference machinery.
![[Pasted image 20260426220905.png|957]]
- In Gas Town, on the MEOW (molecular expression of work) stack, all work is expressed as molecules.
- They can have complex shapes, and loops, and gates, and are in fact Turing-complete, with each step of the workflow executed by a superintelligent AI.
	- AIs understanding TODO lists and acceptance criteria
	- They're reliable at following molecules
	- They get the idea of GUPP
	- They understand that the bureaucracy of checking off issues, no matter how trivial, updates a live activity feed and puts hte work on a permanent ledger.
	- They don't get "bored" and are far less likely to make mistake, because they aren't managing theri own TODO list (except in a single step)

This means that ==molecular owrkflows are durable.==  

IF A MOLECULE IS ON AN AGENT'S HOOK, THEN:
1. The agent is persistent: a Bead backed by git. Sessions come and





