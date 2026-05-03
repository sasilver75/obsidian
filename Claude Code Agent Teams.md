https://code.claude.com/docs/en/agent-teams

Coordinate multiple Claude Code instances working together. 
- One session is the ==LEAD==, and ==TEAMMATES== each run in their own context window and can ==MESSAGE== eachother directly.
- Work distribution: teammates work on different tasks from a shared task list, not the same task. 
- They all share one working directory, ==NO use of [[Git Worktree]]s.== Each teammate is a separate CC session but loads the same project context from the same repo.
	- While they use "file locking" for the task claiming, they also say: "Two teammates editing the same file leads to overwrites. ==Break the work so each teammate owns a different set of files==," so it doesn't seem like it uses file locking for actual edits.
- ==Token cost scale linearly with # of teammates==... in the sense that tokens don't cost any more, you just use more of them, since you've got more teammates.

Start with tasks that have clear boundaries and don't require writing code (e.g. reviewing a PR, researching a library, or investigating a        bug. These show the value of parallel exploration without the coordination challenges that come with parallel implementation).

==Having 5-6 tasks per teammate keeps everyone productive== without excessive context switching. If you have 15 independent tasks, 3 teammates is a good starting point.

Note: ==Split panes require tmux or iTerm2==. The default in-process mode works with any terminal. Split-pane mode isn't supported by VSCode's integrated terminal, Windows Terminal, or Ghostty.


# How it works:
- Architecture
	- Lead
	- Teammates
	- A shared task list
	- A mailbox for inter-agent messaging
- Tasks have pending/in-progress/completed states and can declare dependencies.
- Spawning: Ask the lead in natural language: "Create a team to...", and the Lead picks teammates count (you can specify); teammates can be spawned from existing subagent definitions to reuse roles.
- Coordination: Lead assigns tasks or teammates self-claim (==file-locked== to avoid races). Teammates auto-notify the lead when they're idle.
- Plan approval: Optionally require teammates to plan in read-only mode and get lead sign-off before implementing.


# Setup
- Requires Claude Code v2.1.32+.
  - Enable with ==CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 in env or settings.json==.
  - Display modes: in-process (==Shift+Down to cycle teammates in one terminal==) or tmux/iTerm2 split panes. Default auto.
  - State stored in `~/.claude/teams/{name}/config.json` and `~/.claude/tasks/{name}/` — don't hand-edit.

# Best uses
- Parallel code review with different lenses (e.g. security/performance/tests)
- Competing-hypothesis debugging (adversarial investigators)
- New features split across frontend/backend/tests
- Research with multiple angles

==Avoid for sequential work, same-file edits, or routine tasks -- token cost scales linearly per teammate.==