
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


