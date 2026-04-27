A minimal OSS coding agent created by [[Mario Zechner]]


> "Adapt your coding agent to your workflows, instead of the other way around."


Comes with four packages:
![[Pasted image 20260427014340.png]]
- AI package is a simple abstraction over multiple providers that all pseak differnet transport protocols; can switch between them in the same context/session.
- Agent core is generalized agent loop with tool execution, validation, event streaming ,message queueing, and so forth.
- Terminal interface that's like 600 lines of code and works well
- Coding agent itself, which is both an SDK that you can use in headless mode, or a full terminal use interface coding agent.

![[Pasted image 20260427014456.png]]


![[Pasted image 20260427014540.png]]
You don't need to keep telling agents that they're coding agents; it's 2026 and they know what that is, and it's a big part of their training.


![[Pasted image 20260427014636.png]]
The idea that users need to approve actions... and so then it's safe, is wrong, he says. 
People will either turn that off (yolo mode) or just sit there and type enter whenever a ? comes up.
Containerization is also cnot a complete solution if you're worried about exfiltration of data and prompt injections.

![[Pasted image 20260427014950.png]]
Here's what you can doinstead:
- For MCP use CLI tools plus skills, or build an extension (which we'll see in a bit)
- No subagents, why? Because they're not observable; instead, use tmux and spawn the agent again.
	- You have full control over the agent's inputs/outputs and have full control.
- No plan mode, write to a PLAN.md file instead... can reuse across multiple sessions
- No background bash, dont' need it, we have tmux, same thing
- No built-in todods, write to a TODO.md file.

Or build it using Extensions!

![[Pasted image 20260427015044.png]]
You can give the LLM custom tools that you define; he doesn't think other coding agent harnesses allow for that (just write a simple TS file)
Can write a custom UI
Can bundle it up and put it on npm or git and install it with a single command.
==Everything hot-reloads!== So he develops his own extensions that are project r task specific in Pi, inside hte project; as the agent modifies the extension, he just reloads, and itupdates all the running code.
- This means you can do things like custom compaction (intercept, replace, or augment the default; all the default compaction implementations are not good right now)
Permission gates: Can easily implement them in 50 lines of code and cover what all the other agent harnesses do, if you want that

![[Pasted image 20260427015431.png]]

When Claude Code shipped /btw, it took 5 minutes for someone to replicate that, with more features, in Pi.

`pi-nes`; if you're bored, play a game while the agent is running.
`pi-annotate`: Open the site you're working on, annotate things in the frontend, feed it back into the context
`pi-files-widget`: Live file tracker widget

All of this is extensions, and it takes people usually a couple minutes to an afternoon to build this the way they want it to.

Context is king:
- Tree-structured sessions with branching
	- Your session is a tree, not a 




