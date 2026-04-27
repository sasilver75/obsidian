References:
- [Gas Townhall Documentation (Docs)](https://docs.gastownhall.ai/)

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


![[Pasted image 20260427004447.png]]

![[Pasted image 20260427004451.png]]

![[Pasted image 20260427004509.png]]


![[Pasted image 20260427004533.png]]

![[Pasted image 20260427004544.png]]

![[Pasted image 20260427004609.png]]
![[Pasted image 20260427004624.png]]
![[Pasted image 20260427004635.png]]
![[Pasted image 20260427004701.png]]
![[Pasted image 20260427004724.png]]




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
- Gas Town uses [[Git]] hooks as a propulsion mechanism.
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
- ==NDI, or Nondeterministc Idepotence==, is similar to Temporal's deterministic, durable replay, but Gas Town achieves its durability and guaranteed execution through completely difference machinery.
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
1. The agent is persistent: a Bead backed by git. Sessions come and go, but agents stay.
2. The hook is persistent, and is also a Bead backed by Git.
3. The molecule is persistent; a chain of Beads, also in Git.

So it doesn't matter if Claude Code crashes or runs out of context.  As soon as another session starts up for this agent role, it will start working on that step in the molecule immediately (via GUPP, or when it gets nudged by one of the patrol agents).
- If it founds out that it crashed in the middle of the next step, no biggie, it will figure the right fix, perform it, and move on.

So ==even though the path is fully nondeterministic, the outcome eventually finishes, "guaranteed" as long as you keep throwing agents at it.==

There are tons of edge cases... this description of NDI is over simplifying.


### ==Wisps==: Ephemeral Orchestration Beads

- There are some other corners of our textbook that we should probably touch on...
- Most of the time, you don't care about this stuff, and you care about convoys starting and finishing, watching your activity feeds and dashboards... but Gas Town's molecular "chemistry" has a lot of rich corners that in active use in the orchestration.

==Wisps== are ==ephemeral Beads==. They are in the database, get hashIDs, and act as regular Beads, but ==they are not written to the JSONL file, and thus not persisted to Git!==

At the end of their run, Wisps are "burned" (destroyed). Optionally they can be squashed not a single-line summary digest that's committed to git.

==Wisps are important for high-velocity orchestration workflows.==
- All the patrol agents (Refinery, Witness, Deacon, Polecats) create wisp molecules for every single patrol or workflow run. They ensure that the workflows complete transactionally, but without polluting Git with orchestration noise.


### Patrols
- The ephemeral workflows that run for ==patrol workers==, most notably: Refinery, Witness, Deacon.
- ==A patrol is an ephemeral (wisp) workflow that an agent runs in a loop.==
	- Patrols have exponential backoff: The agent will gradually go to sleep if it finds no work in its patrol steps, by waiting longer and longer to start the next patrol.
	- Any mutating `gt` or `bd` commands will wake the town, or you can do it yourself with the `gt` command, starting up individual workers, groups, a rig, or the whole town.

![[Pasted image 20260426222135.png]]

The ==Refinery=='s patrol is pretty simple. It has some preflight steps to clean up the workspace, then it processes the Merge Queue (MQ) until its empty, or needs to recycle the session.
- It has some post-flight steps in the molecule when it's ready to hand off.
- Soon, there will be plugins that muck with the MQ and try to order it intelligently.

The ==Witness's== patrol is a little more complex; it has to check on the wellbeing of the Polecats, and the Refineries. It also peeks in at the Deacon, just to make sure it's not stick.
- The Witness also runs Rig-level plugins.

The ==Deacon=='s patrol has a lot of important responsibilities:
- Runs Town-level plugins, which can do things like provide entire new UIs or capabilities.
- Also involved in the protocol for `gt handoff` and recycling agent sessions, and ensuring some workers are cleaned up properly.
- The patrol got complex enough, that we added ==Dogs== as helpers, the Deacon's personal crew.
	- It hands complex work and investigations off to Dogs, so that long-running patrol steps don't interfere with the town's core eventing system, which is cooperative and mail-based.

#### ==Plugins==
- GT defines plugins as "coordinated or scheduled attention from an agent."
- Gas town workers run workflows (often in patrol loops), and any workflow can contain any number of "run plugins" steps.

I plan to implement a great deal of functionality in Gas Town as plugins.
 They just didn't make it into the v1 launch. They're probably going to wind up as formulas in the Mol Mall.

![[Pasted image 20260426224647.png|908]]


### Convoys
- Everything is Gas Town, all work, rolls up into a ==Convoy==
	- The Convoy is Gas Town's ticketing or work-order system:

![[Pasted image 20260426224751.png]]

==A Convoy is a special bead that wraps a bunch of work into a unit that you track for delivery.==
- It DOES NOT use the Epic structure, because the tracked issues in a Convoy are not its children. Most of them already have another parent.
- The fundamental primitive for slinging work around in Gas Town  is `gt sling`:
	- If I tell the Mayor "Our tmux sesssions are showing the wrong number of rigs in the status bar, file it and sling it", then the Mayor will file a bead for the problem, then `gt sling` it to a pole cat, which works on it immediately.

I often tell my Beads crew to sling the release molecule to a polecat. The polecat will walk through the 20-step release process, finish it of, and then I'll be notified that the Convoy has landed/finished.

It's confusing to hear: "issue wy-a7je4 just finished", so now ==we wrap every single unit of slung work== (from a single polecat sling to a big swarm someone kicks off) ==with a Convoy==.

==Convoys are basically features==. Whether it's a tech debt cleanup or actual features, or a bug fix, each ==convoy is a ticketing unit of GT's work-order architecture==.
- Note that a Convoy can have multiple swarms "attack" it (work on it) before it's finished. 
- Swarms are ephemeral agent sessions taking on persistent work.
- Whoever is managing the Convoy (e.g. Witness) will keep recycling polecats and pushing them on issues.


## Gas Town Workflow

- The most fundamental workflow in GT is the ==handoff== `gt handoff` or the `/handoff` command, or just say "let's hand off."
- Your worker will optionally send itself work, then restart its session for you, right there in tmux.
- ==All of your workers that you direct==, the Mayor, your Crew, and sometimes the others, ==will require you to let them know it's time to hand off.==
- Other than that, Gas Town dev loop is more or less the same as it is with Claude Code (and Beads), just more of it.
	- You get swarms for free ($)
	- You get decent dashboards
	- You get a way to describe workflows
	- You get mail and messaging
	- ...that's about it.

## Planning in Gas Town
- GT needs lots of fuel: It both consumes and produces guzzoline, or work molecules.
- Aside from just keeping GT on the rails, ==the hardest problem is keeping it fed; it churns through implementation plans so quickly that you have to do a LOT of design and planning to keep the engine fed.==
- On the consumption side, you feed GT epics, issues, and molecules (constructed workflows).
- On the production side, you can use your own planning tool like [Spec Kit](https://github.com/github/spec-kit) or [BMAD](https://docs.bmad-method.org/), and then once your plan is ready, ask an agent to convert it into Beads kits.
- You can use ==formulas== to generate work. If you want every piece of coding work (or design work, or UX work) to go through a particular template or workflow, you can define it as a molecule, and then "wrap" or compose the base work with your orchestration template.
	- Yegge implemented a formula for Emmanuel's "Rule of Five," which is the observation that if you make an LLM review something 5 times, with different focus areas each time through, it generates superior outcomes and artifacts.

This can generate LARGE workflows that can take many hours for days for you to crank through, especially if you are limiting your polecat numbers to throttle your costs or token burn.



# Kubernetes Comparison
- Both coordinate unreliable workers toward a goal.
- ==HERE:== Both have a control plane (Mayor/Deacon vs Kube-Scheduler/Controller-Manager) watching over execution nodes (Rigs vs Nodes), each with a local agent (Witness vs Kubelet) monitoring ephemeral workers (Polecats vs Pods). Both use a source of truth (Beads vs [[etcd]])
	- ==These are apparently natural shapes that emerge when you need to herd cats at scale.==

The big difference is: Kuberentes asks: "Is it running? while Gas Town asks "Is it done?"
- K8s optimizes for runtime (keeping N replicas alive, restarting crashed pods, maintain the desired state forever), while GT optimizes for completion (finish this work, land the convoy, then nuke the worker and move on).
- K8s pods are anonymous cattle; Gas Town polecats are credited workers whose completions accumulate into CV chains, and the sessions cattle.
- K8s reconciles towards a continuous desired state; Gas Town proceeds towards a terminal goal.


______________________

[The Future of Coding Agents, Steve Yegge](https://steve-yegge.medium.com/the-future-of-coding-agents-e9451a84207c) (January 4) [[Steve Yegge]]

- 3 days since Gas Town launch.
- It's currently unstable; it oozes, rather than whirs.
- The instability will fade over the course of 2026; it will go from a slef-propellign slime monster to a shiny, well-run agent factory.
	- Models will get smarter
	- Gas Town and Beads are gong to finally make it into the training corpus
		- Agents use Beads naturally and smoothly with no training, and Gas Town will get there too -- fast!
- The gas town community is going nuts (>50 PRs in 3 days)
- ==Agents will know all about Gas Town by summer.==

> I have been curating Gas Town the same way I did Beads, using the ==Desire Paths approach to agent UX==. You ==tell the agent what you want, watch closely what they try, and then implement the thing they tried. Make it real. Over and over==. Until your tool works just the way agents believe it should work. So Gas Town is gradually becoming agent-friendly, even without being in the training corpus.



> Gas Town eagerly adopted a discovery by [[Jeffrey Emanuel]], author of MCP Agent Mail. He found that combining Mail with Beads led to an ad-hoc “agent village,” where agents will naturally collaborate divide up work and farm it out. Coding agents are pros at email-like interfaces, and you can use mail as an “agent village” messaging system without needing to train or prompt them. They just get it. Gas Town was my attempt to turn an ad-hoc agent village into a coordinated agent town.


> ## Why Golang?
> But I am really liking Go for vibe-coded projects. I probably wrote close to a million lines of code last year, rivaling my entire 40-year career oeuvre to date.
> During the time I was vibe-coding those million lines of code, I learned a lot about what AIs handle well and poorly. And what I found is that models waste a lot of tokens on TypeScript. It’s, like, too much language for them.
> Easily a third to half of all my diffs they created in TS were either complicated type manipulations, or complicated workarounds to avoid having to put proper types on things. Every single “write code” step had to be followed by 2–3 “let’s make it less bad” steps that don’t exist in other languages, to force it to go clean up all its crummy type modeling.
> Python was “fine”. It didn’t suck. It hot reloaded my changes as I was working, which was nice.
> Whereas with Go, every agent has to reinstall and re-codesign the binary locally whenever you make a change, and they tend to forget. The agents don’t waste time wrestling with type modeling. I think for server-side stuff, Python can potentially be great. But for a client-side deployment, it still always felt like a bunch of scripts. I liked Beads’ ability to build and distribute a native Go binary, so I opted for that with Go Gas Town.

> Sure enough, I found on my second major Go project that Go is just… good.

> When the diffs go by in TypeScript, half the time you’re like, what awful thing is my computer up to now? But with Go, it’s just _boring_. It’s writing log files, doing simple loops, doing simple conditionals, reading from maps and arrays, just super duper plain vanilla stuff. Which means you can always understand it! Speaking as someone who has studied and used 50+ programming languages, always looking for elegance and compactness — ==to my surprise, Go is a real boon to vibe-coding systems programmers.==

> Is TypeScript still the best for Web apps? Yeah, probably. I’m just glad I don’t have to build one.


____________


[Software Survival 3.0](https://steve-yegge.medium.com/software-survival-3-0-97a2a6255f7b) (January 28, 2026)[[Steve Yegge]]

> In this post, I’m going to make a prediction about which software will survive, if you believe Karpathy, in a world where AI writes all the software and is essentially infinitely capable. I think you can make a simple survival argument that comes down to selection pressure.

> Karpathy describes a future where AI can build pretty much anything on demand,

> It’s getting easier and easier to build what you need rather than buying it.

> The trajectory is exponential, so home-grown medium-scale SaaS will be on the table by EOY.

> It’s not just SaaS. We’ve already seen entire categories begin to be eaten up by AI: Stack Overflow and Chegg were early victims, but now we’re seeing pressure on new sectors.

# The Selection Argument

> Inference costs tokens, which cost energy, which costs money. For purposes of computing software survival odds, we can think of {tokens, energy, money} all as being equivalent, and all are perpetually constrained.

> This resource constraint, I predict, will create a selection pressure that shapes the whole software ecosystem with a simple rule: ==**software tends to survive if it saves cognition**====.==

> I think that systems have a financial and indeed an ethical obligation to minimize compute costs for solving cognitive problems, because energy is rapidly becoming the world’s biggest constraint.

> But even with a bazillion smaller models deployed everywhere including your dental fillings, I think there is still a role for large classes of “old-fashioned” software systems that aren’t models, and don’t necessarily use AI at all. This essay concerns those systems, which I think will survive and thrive if they have the right properties.

> My hunch is that if a tool saves AIs tokens, it has a high chance of being used and surviving. And tools that don’t save tokens will gradually be phased out.

```
Survival(T) ∝ (Savings × Usage × H) / (Awareness_cost + Friction_cost)
```
![[Pasted image 20260426234553.png]]


## Lever 1: Insight Compression
First lever: your software can make itself useful by compressing insights. The software industry has accumulated a lot of hard-won knowledge that would be expensive to rediscover. Many systems compress hard-won insights into reusable form.

There is no better example than Git. It’s not going anywhere. Sure, it’s not hard for an AI to build its own version control system. But Git’s model–the DAG of commits, refs as pointers, the index, the reflog — I’m already losing you, but that’s the point. Git represents decades of accumulated wisdom about how to track changes when multiple people are working on the same thing, changing their minds, making mistakes, and merging their work back together.

## Lever 2: Substrate Efficiency
As Claude put it, “Nobody is coming for ==grep==.” It’s a great example of a tool that would _also_ be crazy to reinvent, because, like Lever 1, it saves a lot of tokens relative to the effort of using it.

  But grep doesn’t compress any hard-won insights. In fact it’s pretty simple; Ken Thompson famously wrote grep in an afternoon. Grep saves cognition by doing it on a cheaper substrate: CPUs. Algorithmically, it also punches way above its weight class, doing a lot for very little effort. Pattern matching over text is a task where CPU beats GPU by orders of magnitude.

So it would be irrational from any perspective–economic, ecological, moral, or otherwise–to spin up inference to do what grep does. Similarly, LLMs will choose calculators over writing code if they’re available. Tools that enable this lever include parsers, complex transformers like ImageMagick, and many Unix CLI tools.

## Lever 3: Broad Utility
This is the Usage term in the Survival Ratio. It basically amortizes your awareness cost and lowers the threshold for token savings. If you have a truly general-purpose token-saving tool, then it doesn’t really matter if it’s easy for AIs to recreate it. They’ll use the thing that’s everywhere. But how do you make your software be the “obvious” choice for agents?

[[Temporal]] has comparatively high awareness and friction costs, e.g. compared to (say) Postgres, which has been around a lot longer and has much more training data available. But Temporal is as broadly useful as PostgreSQL; just as Postgres can be used to store and query most datasets people care about, Temporal can be used to model and execute most workflows people care about. Temporal has all three levers so far: aggressive insight compression, masterful use of the compute substrate to solve complex problems, and it’s broadly useful. So no AI in its right mind is going to try to clone it for any serious work.

Dolt is another interesting example of software that’s ahead of its time. Gene Kim and I have been saying, “don’t use LLMs for production database access–only use agents in prod when you have Git as a backstop!” Well, what if your database was versioned with Git? Every single change?

[[Dolt]] is OSS that has been around for 8 years, and is only now finally finding its killer app with agent-based prod and devops workflows. With Dolt, agents can make mistakes in prod, and roll back (or forward) with the full power of Git. But they hadn’t solved the awareness problem when I first made Beads, or I’d have used Dolt from the start.

# Lever 4: Publicity
Saving cognition isn’t enough on its own. You also need to solve the awareness problem somehow: the pre-sales problem. Agents have to know about you. Dolt was a great example of a tool with levers 1 to 3 but not 4: I’d have used for Beads sooner if Claude or I had known about it.

# Lever 5: Minimizing Friction
If Awareness is a pre-sales problem, then Product Friction is a post-sales problem. Your agent may be perfectly aware that it has a useful tool, but even a small amount of friction may change its calculation.

Agents always act like they’re in a hurry, and if something appears to be failing for them, they will rapidly switch to trying workarounds. If they’re using your tool and they are having trouble getting it working correctly, they give up super fast.

==Conversely, if you build the tool to their tastes, then agents will use the hell out of it.==
- ((Search "Desire Path" in this file))

...Make your tool intuitive for agents. Getting agents using Beads requires much less prompting, because Beads now has 4 months of “Desire Paths” design, which I’ve talked about before. Beads has evolved a very complex command-line interface, with 100+ subcommands, each with many sub-subcommands, aliases, alternate syntaxes, and other affordances.

# Lever 6: The Human Coefficient
We’ve covered ways to save tokens, and ways to make your tool more palatable to agents. These are great survival strategies starting right now, this year, today. But it’s not the only way.

I think it’s already obvious to everyone that there will be software that thrives not because of token efficiency, but specifically because humans were involved somehow. This software’s value derives from human curation, social proof, creativity, physical presence, approval, whatever. It can be absurdly inefficient because it’s all about that human stink.


__________________

Article: [Vibe Maintainer (Mar 31, 2026)](https://steve-yegge.medium.com/vibe-maintainer-a2273a841040) [[Steve Yegge]]

> Some attendees at an AI Tinkerers meetup in early Feb were asking me what it’s like to be the maintainer of a big OSS project where the community PRs are all AI slop.

> Why is it so important for me to tell you how I deal with a storm of AI-generated PRs? Because I’m beginning to believe that my “vibe maintainer” workflow, crazy as it might sound, will be what a lot of you are doing before long. Everyone who works on successful OSS will soon have to deal with PR storms.

> To give you a sense of the scale I work at, I’m cruising towards 50 contributor PRs a day, combined between [Beads](https://github.com/steveyegge/beads) (20k stars, 5 months old) and [Gas Town](https://github.com/steveyegge/gastown) (13k stars, 3 months old). That’s seven days a week; if I take a day or two off, they pile up and I may have to deal with 100 or more in a single day.

> Through all that, my median time to resolution is about 15 hours, with few PRs waiting more than a few days. This is high velocity. 

> Even with AI help, keeping up with community PRs takes me 15–20 hours a week, usually 2 to 3 hours a day. Sometimes much, much more

> I wish I could tell you it’s easy work. I have at least managed to automate all the easy stuff, which is about half the PRs, sometimes up to two thirds of them. I was recently inspired by ==Dane Poyzer==’s `gt-toolkit` package _—_ a series of Gas Town formulas he published, which now has its own little user subcommunity. Dane’s formulas help you run his ambitious long running idea-to-delivery workflow, which is geared at comprehensive feature development and moving through mountains of work with his Gas Towns. My own PR workflow is now a formula as well.


> Since 99% of my incoming PR submissions are AI-generated, it stands to reason that I could reduce my workload by 99% by saying “No AI PRs.”

> Most OSS maintainers go this route. They straight-up forbid AI-generated or even AI-assisted pull requests. And I can understand why.

> We hear comical stories of once-rejected PRs suddenly being accepted after they’re resubmitted with all the AI DNA scrubbed from the crime scene... But that’s the official party line for most OSS projects today: “No AI.” It all has to be done by sneaking.

> Here’s the problem with that old-school approach. We are headed toward a world in which if you refuse enough PRs, the community will consider you a dead-end street and begin routing around you. They might even develop a community around their version.

> It’s always been trivial to create a fork. The hard part has always been maintaining the fork. It used to take a good-sized team to support a fork, which was almost like a rival gang. Nobody liked a fork. There was always bad blood.
> Today, things are astoundingly different. With the coding agents of 2026, everyone who loves your software is a credible threat to forking you.

> My own approach is radically different from how most OSS maintainers work: I say Yes to AI. Instead of rejecting AI submissions, I encourage everyone to use AI to submit their PRs (subject to a growing list of hygiene rules). Indeed, I both observe and expect 99% of incoming PRs to be AI-assisted.

> Instead of requiring perfect PRs from everyone, I aim to find a quick resolution that is satisfying to all parties. I accept most PRs, but still maintain hard lines on architecture, what goes in core, code quality, and many other AI-era design principles (e.g. [ZFC](https://steve-yegge.medium.com/zero-framework-cognition-a-way-to-build-resilient-ai-applications-56b090ed3e69)). ==If I were to send every PR back to the contributor for fixes, the rest of the community might be losing out on some important fix or feature for days to weeks.== And there it is, sitting there in the PR; it just has issues with it.


> In this situation, if you want to maximize throughput, then you may need to fix the contributor’s code yourself before it can be merged. Most OSS maintainers say, “Go fix your code.” ==I try my best to fix it myself and get it merged. There’s an art to this that I’ll discuss below.==

> My core philosophy is, **help contributors get to the finish line**. I optimize for community throughput. I review every PR and try to find the value in it, and have my worker agents do something appropriate for each one.


(Then he gives an interesting description of some of the techniques and tools he uses)




