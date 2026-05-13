
It's OpenAI's embedding interface for Codex; instead of running Codex as a CLI, you launch `codex app-server` as a subprocess (or connect over [[WebSockets|WebSocket]]) and drive it via [[JSON-RPC]] 2.0.
- Built for IDEs, custom GUIs, and any product that wants to host a Codex agent with full conversational state.
	- Used by the OAI team in their [[OpenAI Symphony]] implementation.
- If you ever wanted to build your own UI around Codex, this is the equivalent of [[Claude Code]]'s internal protocol, a stateful JSON-RPC contract with streaming, approvals, and sandbox integration baked in.


==Three codex surfaces, with different jobs:==
- ==CLI==: Interactive terminal use
- ==SDK==: Non-interactive automation, [[Continuous Integration|CI]] jobs, one-shot calls
- ==App server==: Long-lived, stateful, interactive integrations (think: building a Cursor-style UI on top of Codex)


Protocol:
- [[JSON-RPC]] 2.0, bidirectional (server can push notifications back)
- Transports: `stdio` (newline-delimited JSON, default), [[WebSockets|WebSocket]] (experimental, with auth), or off
- Clients must call `initialize` first to declare themselves
- TypeScript bindings via `codex app-server generate-ts`


What the API exposes:
- ==Threads==: Create/resume/fork/archive/list conversations
- ==Turns==: Start a turn with user input, steer mid-turn, interrupt, stream item/started + item/completed + delta events
- ==Execution  + Approvals==: Runs commands and edits in a sandbox; server emits approval requests when policy requires them
	- This is a bit that the ==SDK doesn't have==
- ==Auth==: API key, ChatGPT [[OAuth]] (browser or device-code), experimental external tokens
- ==Models==: Enumerate with reasoning effort/modalities
- ==Skills, apps (connectors), MCP servers==: Configure and invoke
- ==Filesystem v2==: Watch dirs, read/write, observe changes
- Experimental methods gated behind an `experimentalApi: true` capability flag.





_________________


The interface that [[OpenAI Codex|Codex]] uses to power rich clients (e.g. the Codex VS Code extension). 
- Use when you want deep integration inside your own product,: authentication, conversation history, approvals, and streamed agent events.
- The app-server implementation is [open-source](https://github.com/openai/codex/tree/main/codex-rs/app-server)

## Protocol
- Like [[Model Context Protocol|MCP]], supports bidirectional communication using [[JSON-RPC]] 2.0 messages.
	- Supported transports:
		- [[stdio]]: Default, Newline-delimited JSON ([[JSON Lines|JSONL]])
		- [[WebSockets|WebSocket]]: Experimental, one JSON-RPC message per WebSocket text frame
		- off: Don't expose local transport



# Message Scheme:
Requests include method, params, and id:
```json
{ "method": "thread/start", "id": 10, "params": { "model": "gpt-5.4" } }
```
You can generate a TypeScript schema or a JSON Schema bundle from the CLI.
Each output is specific to the codex version you ran, so the generated artifacts match that version exactly.


Typescript Example:
```js
import { spawn } from "node:child_process";
import readline from "node:readline";

const proc = spawn("codex", ["app-server"], {
  stdio: ["pipe", "pipe", "inherit"],
});
const rl = readline.createInterface({ input: proc.stdout });

const send = (message: unknown) => {
  proc.stdin.write(`${JSON.stringify(message)}\n`);
};

let threadId: string | null = null;

rl.on("line", (line) => {
  const msg = JSON.parse(line) as any;
  console.log("server:", msg);

  if (msg.id === 1 && msg.result?.thread?.id && !threadId) {
    threadId = msg.result.thread.id;
    send({
      method: "turn/start",
      id: 2,
      params: {
        threadId,
        input: [{ type: "text", text: "Summarize this repo." }],
      },
    });
  }
});

send({
  method: "initialize",
  id: 0,
  params: {
    clientInfo: {
      name: "my_product",
      title: "My Product",
      version: "0.1.0",
    },
  },
});
send({ method: "initialized", params: {} });
send({ method: "thread/start", id: 1, params: { model: "gpt-5.4" } });
```

# Core Primitives
- ==Thread==: A conversation between user/Codex agent. Threads contain turns.
- ==Turn==: A single user request and the agent work that follows. Turns contain items and stream incremental updates.
- ==Item==: A unit of input or output (user message, agent message, command runs, file change, tool call and more)

Lifecycle overview:
- Initialize once per connection: Immediately after opening a transport connection, send an `initialize` request with yoru client metadata, then emit `intiialized`...
- Start (or resume) a thread: Call `trhead/start` for a new conversation, `thread/resume` to continue an existing one, or `thread/fork` to branch history into a new thread id.
- Begin a turn: Call `turn/start` with the target `threadId` and user input
- Steer an active turn: Call `turn/steer` to append user input to the currently-in-flight turn without creating a new turn.`
- Stream events: After `turn/start`, keep reading notifications on stdout
	- `thread/archived`, `thread/unarchived`, `item/started`, `item/completed`, `item/agentMessage/delta`, tool progress, and other updates.
- Finish the turn: The server emits `turn/completed` with final status when the model finishes or after a `turn/interrupt` cancellation



