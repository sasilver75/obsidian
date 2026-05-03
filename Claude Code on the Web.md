https://code.claude.com/docs/en/claude-code-on-the-web 

It's Claude Code running in Anthropic-managed cloud VMs (at claude.ai/code) instead of on your laptop. You hand it a task and a GitHub repo; it spins up a sandboxed VM, clones the repo, does the work, and opens a PR. Sessions persist across browser closes and are visible from the Claude mobile app.
- Currently (May 2026) in research preview for Pro / Max / Team / Enterprise users.

Killer features:
- `claude --remote "fix X"` from your terminal kicks off a cloud session and you keep working locally. Run
several in parallel — each is its own VM.
- `claude --teleport` pulls a cloud session (with branch + full conversation history) back into your
terminal to finish locally.
- Auto-fix PRs — Claude subscribes to a PR's CI failures and review comments and pushes fixes
automatically (or asks if ambiguous).
- Plan locally, execute remotely — use plan mode at your desk, commit the plan, then --remote it.
- Mobile monitoring — kick off work from your terminal, check on it from your phone.

# How it works
- Auth to GitHub via the Claude GitHub App (per-repo) or /web-setup (syncs your local gh token).
- Fresh VM per session — Ubuntu 24.04 with Python, Node, Ruby, Go, Rust, Java, Docker, Postgres, Redis, etc. preinstalled. 4 vCPU / 16 GB RAM / 30 GB disk.
- Configuration comes from the repo: CLAUDE.md, .claude/ settings, hooks, agents, commands, .mcp.json. Anything only on your laptop doesn't carry over.
- Setup scripts run once per environment and get cached (~7 days), so package installs are fast on subsequent sessions.
- Network access is locked down by default to a "Trusted" allowlist (package registries, GitHub, cloud SDKs). You can broaden or restrict it.
- GitHub operations go through a proxy — your token never enters the sandbox, and pushes are restricted to the working branch.