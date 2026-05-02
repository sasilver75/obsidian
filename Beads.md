References:
- Medium Blog: [Introducing Beads: A Coding Agent memory system.](https://steve-yegge.medium.com/introducing-beads-a-coding-agent-memory-system-637d7d92514a) (Went through all articles through Apr 2026; here)

A ==lightweight, git-backed issue tracked designed specifically for coding agents==, built by Steve Yegge (ex-
Sourcegraph) in October 2025.

> "It's Google Maps for your plan. Every piece of work is addressable, and it has a label."

The problem it solves:
- Coding agents have no persistent memory between sessions (and every ~10 minutes context window ~resets from compaction).
- Markdown plan files, the default agent approach, fail at scale, because:
	1. Agents keep re-expanding and re-planning the same work, spawning hundreds of plan files.
	2. Plans get lost, orphaned, or contradicted across sessions
	3. Agents near context limits disavow discovered problems, rather than recording them ("That broken test wasn't us!")

Beads:
- ==A specialized issue tracker== where ==work items are structured data, not prose==.
- Key properties:
	- First-class dependencies: 4 link types (==blocking==, ==parent/child==, ==discovered-from==, provenance)
	- `bd ready --json`: Agents query for unblocked work instantly, no text parsing
	- [[JSON Lines|JSONL]] + [[Git]]: Stored as JSONL lines in git, giving both queryability and version history. [[SQLite]] is used as well for queryability.
		- Unlike Github Issues (flat) or Jira (two levels max), Beads supports unlimited parent/child nesting, which examples modeling genuinely complex long-horizon plans.
	- Distributed: Multiple agents on multiple machines can share the same DB via git; AI resolves merge conflicts.


![[Pasted image 20260426172303.png]]

Setup is easy:
- Install the `bd` CLI
- Add one line to `CLAUDE.md` or `AGENTS.md` pointing to it. Agents adopt it spontaneously.
- Core insight is that ==external structured memory with dependency tracking solves the agent amnesia problem far better than any hierarchical markdown plane scheme.==

"Landing the Plane" concept: A structured, end-of-session protocol where the agent syncs work, runs quality gates, manages git state, and writes a handoff prompt before compacting/clearing. 

Yegge explicitly calls Beads a leaky abstraction that requires the user to stay conscious and participatory. This is an honest acknowledgement that it's not a fire-and-forget system, it amplifies agentic workflows rather than automating them.

Yegge mentions that he thinks that his agents write better Go code than TypeScript code.

Yegge says that beads is ==deliberately not a planning tool==: You plan externally (separate tool, conversation, whatever), and then import the resulting epics/issues into Beads. The distinction matters: Beads track _what to do next and what's blocked, not why you're doing it._

`bd doctor` and `bd cleanup` are hygiene rituals... It's recommended to keep the working issue set small (under ~500) and run diagnostics daily. This is the maintenance task of a [[JSON Lines|JSONL]]-backed system: It degrades with size and needs more active pruning.

==Restart agents frequently before tasks==, both as a cost optimization and quality improvement: ==fresh context windows make better decisions.==

==Agent Village== pattern: When you combine Beads (shared memory/issues) with something like [[Agent Mail]] (agent-to-agent messaging), agents autonomously self-organize: One takes leadership, distributes tasks via issues, others pick up ready work. The interesting claim is that this requires no special setup, just both tools being available.

Recurring theme from Yegge: "==keep sessions short, keep issue lists small, restart often.=="


==Rule of Five:== ==Force agents to review their own work 4-5 times before trusting it==. At iteration 4-5, it "converges," and the agent says it can't improve it further. Apply at every stage: design, implementation plan, code, tests. The reasoning is that LLM cognition is breadth-first, so the first output is always a rough sketch; ==most people take the first output and wonder why quality is inconsistent.==

Agent UX is a real design discipline: The --body vs -description example

Deliberately build ahead of current AI capability: Keep a backlog of "too hard for AI today projects," and watch them fall with each model release. It's better to plan for your software to work well when smarter models arrive, not just today's model. This is a different way to think about project sequencing.

==The Merge Wall is the unsolved swarming problem== (2025): Worker D finishing after A+B+C have landed may need a complete redesign, not a rebase; no tool hides this yet. Merge wall is what happens when you try to parallelize agent work and then bring it back together. You ultimately need serialization, forcing rebases to happen one at a time, giving each worker enough context to understand what changed before them and adapt their work accordingly. Some work is inherently serial. ==Knowing when to swarm and when to serialize is most of the skill in agent orchestration.==

Yegge argues that you should spend ==30-40% of your vibe coding time not on features, but on dedicated code health sessions== where you explicitly ask agents to hunt for problems: oversized files, redundant systems, dead code, misplaced files. ==Everything found gets filed as an issue rather than fixed immediately.== 
- The reason a number if so high is that agents have no memory between sessions... the cleanliness of the codebase directly determines the quality of every agent decision - a messy codebase doesn't just feel bad, it actively makes agents slower and dumber on every task.

Yegge argues Joel Spolsky's famous "never rewrite your software" rule — which held for 25 years — now has an expiration date, and that ==software is now throwaway: expect < 1 year shelf life.==
- Joel Spolsky: The idea that new code is better than old is patently absurd: old code has been used, it has been tested. Lots of bugs have been found, and they've been fixed.
- Yegge: We're entering a surprising new phase of SWE in which rewriting things is often easier and smarter than trying to fix them.
	- You'll do a big refactoring via agents, and all tests will be broken, and the agents inevitably struggle to fix them.
	- If you delete the tests and make new ones... it happens faster, the new tests are great, had great coverage, etc... and most importantly it was faster/cheaper. With new tests, it can focus just on the new behavior.

Agent UX for tool design is important: During development agents would use `--body` instead of `--description` when filing an issue, which would fail; Because the models were largely trained on GH Issues, and GHI's cli tool uses `--body` for filing issues. Models love to reach for the familiar (e.g. tools like `grep`).