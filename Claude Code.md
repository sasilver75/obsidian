A coding agent from [[Anthropic]]


Claude can act, read your code, edit files, run commands, search the web, and interact with extenral services.

The built-in tools generally fall into FIVE categories:
1. ==File operations==: Read files, edit code, create new files, rename, and organize
2. ==Search==: Find files by pattern, search content with regex, explore codebases
3. ==Execution==: Run shell commands, start servers, run tests, use git
4. ==Web==: Search the web, fetch docs, look up error messages
5. ==Code intelligence==: See type errors and warnings after edits, jump to definitions, find references

Claude also has tools for spawning ==subagents==, asking you questions, and other orchestration tasks.
- See the [Tools Reference](https://code.claude.com/docs/en/tools-reference) for a full list

When you say something like `"Fix the failing tests!"`, Claude might:
1. Run the test suite to see what's failing.
2. Read the error output.
3. Search for the relevant source files
4. Read those files to understand the code
5. Edit the files to fix the issue
6. Run the tests again to verify

The built-in tools are the foundation, but you can extend what Claude knows with ==[[Skill|Skills]]==, connect to external services with [[Model Context Protocol|MCP]], automate workflows with ==Hooks==, and offload tasks to ==subagents==.
- These extensions form a layer on top of the core agentic look.
- See Extend [Claude Code](https://code.claude.com/docs/en/features-overview) for more

The ==[[Language Server Protocol|LSP]]== tool gives Claude code intelligence from a running language server. After each file edit, it automatically reports type errors and warnings, so Claude can fix issues without a separate build step.
- Claude can also call it directly to navigate code (jump to symbol definitions, find all references to a symbol, trace call hierarchies, etc.)

The ==Monitor== tool lets Claude watch something in the background and react when it changes, without pausing conversation. Ask Claude to:
- Tail a log file and flag errors as they appear.
- Poll a PR or CI job and report when its status changes.
- Watch a directory for file changes.
- Track output from any long-running script you point at it.
You just keep working in the same session, and Claude interjects when an event lands. You can stop a monitor by asking Claude to cancel it or by ending the session.

You can check which tools are available by simply asking Claude:
```
What tools do you have access to?
```
And Claude gives a conversational summary!


# CLI Reference
- See [CLI reference](https://code.claude.com/docs/en/cli-reference)
Notable ones:
```bash
claude -c — continue last conversation here
claude -r "<name>" / -n "<name>" — resume / name sessions
claude -w <name> — isolated git worktree

claude -p "query" — one-shot print mode

claude --output-format json / --json-schema — structured output
```

# Commands Reference
- See [Commands Reference](https://code.claude.com/docs/en/commands)
- ==Commands== control claude from *inside a session!* They provide a quick way to switch models, manage permissions, clear context, run a workflow, and more.
	- Type `/` to see every command, and type to filter.
Notable ones:
```
/clear : Start fresh; old conversation still in /resume
/compact : summarize to free context but keep going
/context : visualize what's eating your context window
/rewind : roll the conversation back to a previous turn
/branch : Branch the conversation;switch back via /resume
/resume [session] : pick up a past session by ID or name
/plan [description] : jump into plan mode, optionally with a starting task
/review [PR] : local PR review; /ultrareview for the deep cloud version
/simplify [focus] : a multi-agent review of recent changes, then fixes
/security-review : scans pending diff for vulns
/model [model] : switch model
/effort [level] : low/medium/high/xhigh/max
/config : theme, model, output style
/memory : Edit CLAUDE.md and manage auto-memory
/hooks, /mcp, /agents, /skills : inspect the respective configs
/skills : list available skills; press t to sort by token counts
/init : initialzes project with a CLAUDE.md guide... Set CLAUDE_CODE_NEW_INIT=1 for an interactive flow
```


# Environment Variables
- See [Environment Reference](https://code.claude.com/docs/en/env-vars)
- Set these before launching `claude`, or configure them in `settings.json` under the `env` key to apply them to every session, or to roll out across your team.
Notable ones:
```
ANTHROPIC_MODEL — default model
CLAUDE_CODE_SUBAGENT_MODEL — model used by subagents
CLAUDE_CODE_EFFORT_LEVEL — low/medium/high/xhigh/max/auto
```


# Tools Reference
- See [Tools Reference](https://code.claude.com/docs/en/tools-reference)
- Claude has access to a set of built-in tools that help it understand and modify your codebase.
- To disable a tool entirely, add its name to the `deny` array in your permission settings.
- To add custom tools, connect an [[Model Context Protocol|MCP Server]], and to extend Claude with reusable prompt-based workflows, write a  ==skill==, which runs through the existing `Skill` tool rather than adding a new tool entry.
```
File and Search:
	Read
	Edit
	Write
	Glob: Pattern-match filenames
	Grep: Search file contents
	
Execution
	Bash: Shell commands; env vars don't persist across calls
	Monitor: Watch a long-running script and feed output lines back as events
	
Code Intelligence
	LSP: Jump to definition, find references, etc.

Web
	WebFetch: fetch a URL
	WebSearch: web search

Agents and Planing
	Agent: spawn a subagent with its own context window
	EnterPlanMode / ExitPlanMode
	EnterWorkTree / ExitWorktree
	AskUserQuestion: Multiple-choice clarifying questions
	
Tasks and scheduling
	TaskCreate / TaskUpdate / TaskList / TaskGet: Interactive session task list
	CronCreate / CroSList / CronDelete: schedule prompts within a session
	TaskStop: Kill a background task

Other
	Skill: Run a skill in the main conversation
	ToolSearch: Load deferred tools on demand (when MCP tool search is on)
	ListMcpResourcesTool / ReadMcpResourcesTool: browse/read MCP resources
```
Above: These tool names are the ones you'd use in /permissions rules, subagent tool lists, and hook matchers. 

# Interactive Mode
- See keybindings here for [Interactive Mode](https://code.claude.com/docs/en/interactive-mode)


# Checkpointing
- CC automatically tracks Claude's file edits as you work, allowing you to quickly undo changes and rewind state if anything gets off track.
- As you work with Claude, checkpointing automatically captures the state of your code before each edit, which lets you pursue ambitious, wide-scale tasks knowing you can always return.
- Use the `/rewind` command to open the rewind menu, select the point you want to act on, then choose:
	- ==Restore code and conversation==: Revert both code and conversation to that point
	- ==Restore conversation==: Rewind to that message while keeping current code
	- ==Restore code==: Revert file change while keeping the conversation
	- ==Summarize from here==: Compress convo from this point forward into summary, freeing context window space
		- MESSAGES FROM BEFORE THE TARGETED POINT STAY INTACT!
	- Nevermind

The three Restore options revert state, undoing code changes, conversation history, or both.

The Summarize from here one just replaces context with compact AI-generated summary of messages from selected point, but no files on disk are changed. It's similar to /compact, but instead of summarizing the entire conversation, you keep EARLY context in full detail, and only compress parts that are using up space. 
- Useful for
	- Exploring alternatives
	- Recovering from mistakes
	- Iterating on features
	- Freeing context space

# Hooks Reference
- [Hooks Reference Link](https://code.claude.com/docs/en/hooks)
	- See a table here that summarize when each event fires.
-  Hooks are ==user-defined shell commands, HTTP endpoints, or LLM prompts== that execute automatically at specific points in CC's lifecycle.
- Hooks are defined in JSON setting files, with the configuration having three levels of nesting:
	1. Choose a hook event to respond to (e.g. `PreToolUse`)
	2. Add a matcher group to filter when it fires, like "only for the bash tool"
	3. Define one or more hook handlers to run when it's matched.
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/block-rm.sh"
          }
        ]
      }
    ]
  }
}
```
And the script reads hte JSON input from stdin, extracts the command, and returns a permissionDecision of "deny" if it contains rm -rf:
```python
#!/bin/bash
# .claude/hooks/block-rm.sh
COMMAND=$(jq -r '.tool_input.command')

if echo "$COMMAND" | grep -q 'rm -rf'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Destructive command blocked by hook"
    }
  }'
else
  exit 0  # allow the command
fi
```
Say CC decides to run `Bash "rm -rf /tmp/build"`:
![[Pasted image 20260427214943.png]]


![[Pasted image 20260427214812.png]]



# Plugins Reference
- [Reference](https://code.claude.com/docs/en/plugins-reference)
- A ==Plugin== is a ==self-contained directory of components that extends Claude Code== with custom functionality.
	- ==They're bundles of: skills, agents, hooks, MCP servers, LSP servers, and monitors.==
- Location: `skills/` or `commands/` directory in plugin root
- Skills are directories with `SKILL.md`;  commands are simple markdown files.
```
skills/
├── pdf-processor/
│   ├── SKILL.md
│   ├── reference.md (optional)
│   └── scripts/ (optional)
└── code-reviewer/
    └── SKILL.md
```
- Skills and commands are automatically discovered hwen teh plugin is installed, and Claude can invoke them automatically based on task context.
- Plugins can also provide specialized subagents for specific tasks that Claude can invoke automatically when appropriate:
	- Location: `agents/` directory in plugin root
- Agents are defined as markdown files describing agent capabilities:
```
---
name: agent-name
description: What this agent specializes in and when Claude should invoke it
model: sonnet
effort: medium
maxTurns: 20
disallowedTools: Write, Edit
---

Detailed system prompt for the agent describing its role, expertise, and behavior.
```
- Agents appear in the `/agents` interface, and Claude can invoke agents automatically based on task context.
- Plugin agents work alongside built-in Claude agents.
- Plugins can provide event handlers that respond to Claude Code events autonomously.
- Location: `hooks/hooks.json` in plugin root, or inline in plugin.json
- They're JSON configuration files with even matchers and actions:
```
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "${CLAUDE_PLUGIN_ROOT}/scripts/format-code.sh"
          }
        ]
      }
    ]
  }
}
```
Plugin hooks respond to the same lifecycle events as user-defined hooks
- Plugins can also bundle MCP Servers to connect CC with external tools and services.
- Location: .mcp.json in plugin root, or inline in plug.json
- Format: Standard MCP server configuration
```
{
  "mcpServers": {
    "plugin-database": {
      "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",
      "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
      "env": {
        "DB_PATH": "${CLAUDE_PLUGIN_ROOT}/data"
      }
    },
    "plugin-api-client": {
      "command": "npx",
      "args": ["@company/mcp-server", "--plugin-mode"],
      "cwd": "${CLAUDE_PLUGIN_ROOT}"
    }
  }
}
```
Plugins can also provide [[Language Server Protocol|LSP]] servers to give Claude real-time code intelligence (Code navigation, language awareness).
Plugins can declare background ==monitors== that CC starts automatically when the plugin is active. Each monitor runs a shell command for the life of the session and delivers each stdout line to Claude as a notification, so Claude can react to log entries, status changes, or polled events without being asked to start the watch itself.

A complete plugin follows this structure:
```
enterprise-plugin/
├── .claude-plugin/           # Metadata directory (optional)
│   └── plugin.json             # plugin manifest
├── skills/                   # Skills
│   ├── code-reviewer/
│   │   └── SKILL.md
│   └── pdf-processor/
│       ├── SKILL.md
│       └── scripts/
├── commands/                 # Skills as flat .md files
│   ├── status.md
│   └── logs.md
├── agents/                   # Subagent definitions
│   ├── security-reviewer.md
│   ├── performance-tester.md
│   └── compliance-checker.md
├── output-styles/            # Output style definitions
│   └── terse.md
├── themes/                   # Color theme definitions
│   └── dracula.json
├── monitors/                 # Background monitor configurations
│   └── monitors.json
├── hooks/                    # Hook configurations
│   ├── hooks.json           # Main hook config
│   └── security-hooks.json  # Additional hooks
├── bin/                      # Plugin executables added to PATH
│   └── my-tool               # Invokable as bare command in Bash tool
├── settings.json            # Default settings for the plugin
├── .mcp.json                # MCP server definitions
├── .lsp.json                # LSP server configurations
├── scripts/                 # Hook and utility scripts
│   ├── security-scan.sh
│   ├── format-code.py
│   └── deploy.js
├── LICENSE                  # License file
└── CHANGELOG.md             # Version history
```

Install plugins from available marketplaces: `claude plugin isntall <plugin> [options]`
Remove plugins: `claude plugin uninstall <plugin> [options]`
Enable a disabled plugin `claude plugin enable <plugin> [options]`
Disable an enabled plugin without uninstalling it: `claude plugin disable <plugin> [options]`
Plugin update: `claude plugin update <plugin> [options]`
List plugins: `claude plugin list [options]`
Plugin tag: `claude plugin tag [options]`


# Skills
- [Reference](https://code.claude.com/docs/en/skills#bundled-skills)
- Skills extend what Claude can do. Create a `SKILL.md` file with instructions, and Claude adds it to its toolkit. Claude uses skills when relevant, or you can invoke one directly with `/skill-name`.
- Bundled skills:
	- CC includes a set of bundled skills available in every session, including `/simplify`, `/batch`, `/debug`, `/loop`, and `/claude-api`. 
	- Unlike most built-in commands which executed fixed logic directly, bundled skills are prompt-based, giving Claude a detailed playbook and let it orchestrate the work using its tools. You invoke them the same way as any other skill, by typing `/` followed by the skill name.


To create a skill:
1. Create the skill directory in your ==personal skills== folder; personal skills are available across all your projects.
```
mkdir -p ~/.claude/skills/explain-code
```
2. Write the `SKILL.md` file with two parts: [[Yet Another Markup Langauge|YAML]] frontmatter (between `---` markers) that tells Claude when to use the skill, and [[Markdown]] content with instructions Claude follows when the skill is invoked. The `name` field becomes the `/slash-command`, and the description helps Claude decide when to load it automatically.
```markdown
---
name: explain-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```
YAML frontmatter only requires:
- `   name` - A unique identifier for your skill (lowercase, hyphens for spaces)
- `description` - A complete description of what the skill does and when to use it
3. Test the skill!
	- Let Claude invoke it automatically by asking something that matches its description `How does this code work?`
	- OR Invoke it directly with the skill name: `/explain-code src/auth/login.ts`

Where you store a skill determines who can use it:
![[Pasted image 20260427222829.png]]
- When skills share the same name across levels, enterprise overrides personal, and personal overrides project.
- Plugin skills use a `plugin-name:skill-name` namespace, so they don't conflict with other levels.
- CC watches skill directories for file changes, so ==adding/editing/removing a skill== under `~/.claude/skills/` or the project `.claude/skills/` ==takes effect within the current session without restarting!==

Each skill is a directory with `SKILL.md` as the entrypoint:
```
my-skill/
├── SKILL.md           # Main instructions (required)
├── template.md        # Template for Claude to fill in
├── examples/
│   └── sample.md      # Example output showing expected format
└── scripts/
    └── validate.sh    # Script Claude can execute
```
- The `SKILL.md` contains the main instructions, and is required. Other files are optional and let you build more powerful skills: templates for Claude to fill in, example outputs showing expected format, scripts claude can execute, or detailed reference documentation. 
- Reference these files from your `SKILL.md` so that Claude knows what they contain and hwen to load them.

Types of Skills Content:
- Reference content ==adds knowledge== Claude applies to your current work. ==Conventions, patterns, style guides, domain knowledge==. This content runs inline so Claude can use it alongside your conversation context.
	- ((Why should something like "API design patterns for this codebase" be a skill instead of in CLAUDE.md?))
- Task content gives Claude ==step-by-step instructions for a specific action, like deployments, commits, or code generation==. These are often actions you want to invoke directly with /skill-name rather than letting Claude decide when to run them. Add disable-model-invocation: true to prevent Claude from triggering it automatically.
	- Often actions you want to invoke directly with `/skill-name` rather than letting Claude invoke them itself. You can add a`disable-model-invocation: true` to the YAML header to prevent Claude from triggering it automatically.
		- Alternatively, you can use `user-invocable: false` so that only Claude can invoke the skill; use thies for background knowledge that isn't actionable as a command.

Skills support string substitution for dynamic values in the skill content:
![[Pasted image 20260427225127.png]]

You can also configure sill behavior using YAML fields in the header of your SKILL file:
![[Pasted image 20260427225100.png]]


## Running skills in a subagent
- Add `context: fork` to your skill's YAML frontmatter when you want a skill to run in isolation. The skill content becomes the prompt that drives the subagent. It won’t have access to your conversation history.
- ![[Pasted image 20260427225302.png]]

Example: Research skill using Explore agent:
- This skill runs research in a forked Explore agent:
```
---
name: deep-research
description: Research a topic thoroughly
context: fork
agent: Explore
---

Research $ARGUMENTS thoroughly:

1. Find relevant files using Glob and Grep
2. Read and analyze the code
3. Summarize findings with specific file references
```
- When this runs
	- A new isolated context is created
	- The subagent receives the skill content as its prompt
	- The `agent` field determines the execution environment (model, tools, permissions)
		- There are some built-in agents (Explore, Plan, general-purpose), or any custom subagent from `.claude/agents/`. If omitted, uses `general-purpose`.
	- Results are summarized and returned to your main conversation.

# Claude.md
[`CLAUDE.md`](https://code.claude.com/docs/en/memory) is a markdown file you add to your project root that Claude Code reads at the start of every session. Use it to set coding standards, architecture decisions, preferred libraries, and review checklists. Claude also builds [auto memory](https://code.claude.com/docs/en/memory#auto-memory) as it works, saving learnings like build commands and debugging insights across sessions without you writing anything.

![[Pasted image 20260427230629.png]]

![[Pasted image 20260427230646.png]]

![[Pasted image 20260427230702.png]]


**Size**: target under 200 lines per CLAUDE.md file. Longer files consume more context and reduce adherence.


Specificity: write instructions that are concrete enough to verify. For example:
- “Use 2-space indentation” instead of “Format code properly”
- “Run npm test before committing” instead of “Test your changes”
- “API handlers live in src/api/handlers/” instead of “Keep files organized”

**Structure**: use markdown headers and bullets to group related instructions. Claude scans structure the same way readers do: organized sections are easier to follow than dense paragraphs.

CLAUDE.md files can import additional files using `@path/to/import` syntax. Imported files are expanded and loaded into context at launch alongside the CLAUDE.md that references them.

For private per-project preferences that shouldn’t be checked into version control, create a `CLAUDE.local.md` at the project root. It loads alongside `CLAUDE.md` and is treated the same way. Add `CLAUDE.local.md` to your `.gitignore` so it isn’t committed; running `/init` and choosing the personal option does this for you.


# AGENTS.md
Claude Code reads `CLAUDE.md`, not `AGENTS.md`. If your repository already uses `AGENTS.md` for other coding agents, create a `CLAUDE.md` that imports it so both tools read the same instructions without duplicating them. You can also add Claude-specific instructions below the import. Claude loads the imported file at session start, then appends the rest:

```md
@AGENTS.md

## Claude Code

Use plan mode for changes under `src/billing/`.
```


Claude Code reads CLAUDE.md files by walking up the directory tree from your current working directory, checking each directory along the way for `CLAUDE.md` and `CLAUDE.local.md` files. This means if you run Claude Code in `foo/bar/`, it loads instructions from `foo/bar/CLAUDE.md`, `foo/CLAUDE.md`, and any `CLAUDE.local.md` files alongside them.

All discovered files are concatenated into context rather than overriding each other. Within each directory, `CLAUDE.local.md` is appended after `CLAUDE.md`, so when instructions conflict, your personal notes are the last thing Claude reads at that level.

Block-level HTML comments (`<!-- maintainer notes -->`) in CLAUDE.md files are stripped before the content is injected into Claude’s context. Use them to leave notes for human maintainers without spending context tokens on them. Comments inside code blocks are preserved. When you open a CLAUDE.md file directly with the Read tool, comments remain visible.


