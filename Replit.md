
Replit is a browser-based software development platform that combines an online IDE, AI app-building agents, hosted workspaces, package/runtime management, databases, storage, authentication integrations, and one-click deployments. In the current product, the central unit is a **Project**. A project can contain one or more **Apps** or **Artifacts**, and those artifacts can be previewed, iterated on, and published.

This note is a first-principles tutorial and practical reference for building with Replit. It reflects official Replit documentation checked on 2026-06-03. Replit has changed meaningfully in recent months, so older tutorials that focus only on "Repls" or older always-on workflows may not match the current UI and deployment model.

## Table of Contents

- [Mental Model](#mental-model)
- [What Replit Is Good For](#what-replit-is-good-for)
- [Core Vocabulary](#core-vocabulary)
- [Account, Workspace, and Project Basics](#account-workspace-and-project-basics)
- [The Replit Editor](#the-replit-editor)
- [The AI Workflow](#the-ai-workflow)
- [Prompting Replit Agent](#prompting-replit-agent)
- [Project Setup](#project-setup)
- [Configuration Files](#configuration-files)
- [Working With Code](#working-with-code)
- [Secrets and Environment Variables](#secrets-and-environment-variables)
- [Databases](#databases)
- [App Storage](#app-storage)
- [Authentication](#authentication)
- [Previewing and Debugging](#previewing-and-debugging)
- [Publishing and Deployments](#publishing-and-deployments)
- [Domains, Sharing, and Access Control](#domains-sharing-and-access-control)
- [Complete Project Walkthrough](#complete-project-walkthrough)
- [Testing and Quality](#testing-and-quality)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)
- [Sources](#sources)

## Mental Model

Replit is easiest to understand as four layers:

1. **A cloud development environment**: code editor, shell, package manager, preview window, debugger-style tools, and collaboration all run in the browser.
2. **An AI builder**: Replit Agent can plan, generate, edit, run, debug, and deploy applications with natural-language instructions.
3. **A managed runtime and resource layer**: Replit provides compute, ports, environment variables, databases, object storage, and workflows.
4. **A deployment platform**: completed apps can be published as static sites, autoscaled services, reserved VM deployments, scheduled jobs, or private deployments depending on what the app needs.

The key shift from older mental models: Replit is not only a browser IDE for small scripts. It is now also an AI-assisted app platform where Agent and Assistant can build and maintain complete apps, and where a project can contain multiple artifacts.

## What Replit Is Good For

Replit is strong when you want:
- Fast prototypes, Small to medium full-stack apps, Browser-only development with no local setup, Shareable demos..
Replit can be less ideal when you need:
- Fine-grained infrastructure control, Very large monorepos, Heavy GPU training workloads, Strict enterprise deployment pipelines that must run entirely outside Replit.

## Core Vocabulary

- **Project**: A workspace containing code, configuration, resources, and one or more apps or artifacts.
- **App**: A runnable application inside a project. In common usage, people still say "Replit app" or "Repl", but the current docs increasingly describe projects and publishable artifacts.
- **Artifact**: A publishable output of a project. With Agent 4, Replit supports multiple apps or artifacts in one project, such as a main web app, an admin tool, a mobile app prototype, or a simple game.
- **Replit Agent**: The AI builder that can plan, create, modify, run, debug, and deploy apps from prompts.
- **Assistant**: The coding assistant used inside the workspace for targeted edits, explanations, debugging help, and code changes.
- **Plan Mode**: An Agent mode for discussing and designing what should be built before edits are made.
- **Build Mode**: An Agent mode for applying changes, generating code, running commands, and iterating.
- **Checkpoint**: A saved project state. You can use checkpoints to recover from bad generations or changes.
- **Preview**: The browser-accessible running view of your app inside the Replit workspace.
- **Shell**: The terminal inside the workspace.
- **Console**: Runtime output for the running app.
- **Secrets**: Encrypted environment variables, such as API keys and database URLs.
- **Deployment**: A published production version of an artifact.
- **App Storage**: Replit's object storage product for storing files and objects, formerly referred to as Object Storage in older docs.

## Account, Workspace, and Project Basics

### Creating a Project

You can usually start in one of several ways:

- Prompt Agent to build an app.
- Import a GitHub repository.
- Start from a template.
- Create an empty project and write code manually.
- Remix or fork an existing shared project, if available.

When starting with Agent, begin with a concise description of the outcome:

```text
Build a task tracker for a small team. It should have login, projects,
tasks with due dates, status filters, a dashboard, and a clean responsive UI.
Use a database so data persists after reloads.
```

When starting manually, decide the stack first:

- Python Flask or FastAPI.
- Node.js Express.
- React/Vite.
- Next.js.
- Static HTML/CSS/JS.
- Streamlit.
- A language-specific template such as Go, Rust, Java, or C++.

### Project Lifecycle
A project usually moves through these stages:

1. **Idea**: describe the goal.
2. **Plan**: outline features, data model, and screens.
3. **Build**: generate or write code.
4. **Run**: start the app and inspect preview.
5. **Debug**: fix runtime errors, UI issues, or data bugs.
6. **Persist**: add database, storage, secrets, and auth.
7. **Polish**: improve UX, tests, performance, and copy.
8. **Deploy**: publish an artifact.
9. **Operate**: monitor logs, update code, and redeploy.

## The Replit Editor
The editor is the center of manual development. It usually includes:
- File tree.
- Code editor.
- Shell.
- Console.
- Preview.
- Agent or Assistant panel.
- Package/dependency tools.
- Secrets/resource panes.
- Deployment tools.

### File Tree
Use the file tree to inspect generated code. Agent can create a lot quickly, so keep a habit of reviewing:
- Entry point files.
- Package manifests.
- Database/schema files.
- Route handlers.
- Frontend components.
- Config files.
- Deployment configuration.

### Shell
The Shell is where you run commands:
```bash
npm install
npm run dev
npm test
python -m pip install -r requirements.txt
python main.py
pytest
```

Use the Shell when you need deterministic feedback. Agent is useful, but command output is still the ground truth for build and runtime failures.

### Console
The Console shows app runtime output. Use it to inspect:
- Server startup messages.
- Stack traces.
- Log statements.
- Port binding errors.
- Database connection failures.
- Missing environment variables.

### Preview
The Preview shows the running app. Use it like a real user:
- Click through every main workflow.
- Try empty states.
- Try invalid inputs.
- Reload the page.
- Check mobile sizes if the preview supports it.
- Confirm data persists.

If preview is blank, check the Console before asking Agent for broad changes.

## The AI Workflow
Replit has two related AI workflows:
- **Agent** for larger app-building tasks.
- **Assistant** for more focused help inside code.

### Agent Modes
Current docs describe four Agent performance modes:
- **Lite**: simplest and cheapest mode for small changes.
- **Economy**: balanced mode for routine app work.
- **Power**: stronger mode for complex development.
- **Turbo**: most capable mode for difficult, multi-step work.

Agent also uses:
- **Plan Mode**: discuss and refine a plan before changes.
- **Build Mode**: make changes and run work.

Practical usage:
- Start in Plan Mode for non-trivial apps.
- Switch to Build Mode only after the plan is specific enough.
- Use Lite/Economy for small edits.
- Use Power/Turbo for larger apps, broken builds, or architecture changes.

### Agent 4 and Multiple Artifacts
Official docs say Agent 4 launched in March 2026 and introduced multi-app or multi-artifact support inside one project. That matters because a single project can now represent more than one publishable thing.
Examples:
- A marketing site plus an admin dashboard.
- A web app plus a prototype mobile app.
- A backend API plus a simple frontend client.
- A game plus a level editor.

Do not assume every generated file belongs to one monolithic app. Ask Agent to name and organize artifacts clearly.

### Checkpoints
Checkpoints are one of the most important safety tools. Use them when:
- The app reaches a working milestone.
- You are about to ask for a large change.
- Agent proposes a risky rewrite.
- You want to compare two directions.

Practical habit:
```text
Before the next major change, create a checkpoint named:
"working-auth-and-task-crud"
```
If a generation breaks the app, revert to the last known good checkpoint instead of trying to patch random symptoms.

### Task System
Replit's task system helps Agent break larger work into steps. Use it when you have multiple features:
```text
Create tasks for:
1. Add the database schema.
2. Build authentication.
3. Build the dashboard UI.
4. Add task CRUD.
5. Add filters and search.
6. Add tests.
7. Prepare deployment.

Do not start building until I approve the task list.
```
This keeps Agent from jumping straight into a broad implementation with hidden assumptions.

## Prompting Replit Agent
Good Replit prompts are concrete, scoped, and testable.
### Bad Prompt
```text
Make me a SaaS app.
```
This is too vague. Agent must invent the user, product, data model, stack, layout, and success criteria.

### Better Prompt
```text
Build a small SaaS dashboard for independent consultants to track client invoices.

Core workflows:
- Sign up and log in.
- Create clients.
- Create invoices for a client.
- Mark invoices as draft, sent, paid, or overdue.
- Filter invoices by status.
- Show dashboard cards for total outstanding and paid this month.

Use a clean responsive UI, a real database, and server-side validation.
Before building, propose the stack, data model, and route structure.
```
This gives Agent enough structure to make fewer bad guesses.

### Prompt Template for a New App
```text
I want to build [type of app] for [target users].

The app should solve this problem:
[problem]

Required workflows:
- [workflow 1]
- [workflow 2]
- [workflow 3]

Data that must persist:
- [entity 1 with fields]
- [entity 2 with fields]

Constraints:
- Use [preferred stack] if reasonable.
- Use a database for persistent data.
- Keep the UI [visual style].
- Make it responsive.
- Add basic tests for critical logic.

First, create a plan with:
1. App architecture.
2. Data model.
3. Main screens.
4. API routes or server actions.
5. Deployment assumptions.

Wait for my approval before building.
```

### Prompt Template for Debugging
```text
The app fails when I [exact action].

Observed behavior:
[what happens]

Expected behavior:
[what should happen]

Console or error output:
[paste relevant stack trace]

Please diagnose the root cause first. Then make the smallest fix and run the app
or tests to verify it.
```

### Prompt Template for UI Iteration
```text
Improve the [screen/component] UI.

Keep the current functionality.
Make it easier to scan by:
- Reducing visual clutter.
- Grouping related controls.
- Making primary actions obvious.
- Improving empty and loading states.

Do not change the data model or backend routes unless required.
```

### Prompt Template for Production Readiness
```text
Review this app for production readiness.

Check:
- Secrets and environment variables.
- Database persistence.
- Authentication and authorization.
- Error handling.
- Input validation.
- Deployment settings.
- Logs and health checks.
- Obvious security issues.

Give me a prioritized checklist before making changes.
```

### Prompting Rules
- Ask for a plan before large changes.
- State the target user.
- State the main workflows.
- State persistence requirements.
- Say what should not change.
- Paste exact error output.
- Ask Agent to verify with commands.
- Use checkpoints before major rewrites.
- Prefer small iterations after the first build.

## Project Setup
Project setup depends on the stack. Replit can infer many templates, but you should still know what the pieces mean.

### Node.js App
Typical files:
```text
package.json
src/
  index.js
```

Example `package.json`:
```json
{
  "scripts": {
    "dev": "vite --host 0.0.0.0",
    "start": "node src/index.js",
    "test": "vitest"
  },
  "dependencies": {
    "express": "^4.19.2"
  },
  "devDependencies": {
    "vite": "^5.0.0",
    "vitest": "^1.0.0"
  }
}
```

For web servers, bind to `0.0.0.0` and use the port expected by the environment.

### Python App
Typical files:
```text
main.py
requirements.txt
```

Example `requirements.txt`:
```text
fastapi[standard]
uvicorn
```

Example server:
```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "hello from Replit"}
```

Run command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Static Site
Typical files:
```text
index.html
style.css
script.js
```
Static sites are the simplest deployment type when you do not need a backend.

### Full-Stack App
A full-stack app often has:
```text
package.json
server/
client/
shared/
database/
```

or:
```text
app/
  main.py
  routers/
  models.py
frontend/
```

Ask Agent to explain the generated structure. Do not treat generated folder names as magic.

## Configuration Files
Replit app configuration commonly involves two hidden files:
- `.replit`: controls run commands, entrypoints, deployment commands, port mapping, packager behavior, and other app behavior.
- `replit.nix`: controls system-level dependencies through Nix.
These files may be hidden by default in the file tree. Use the file tree menu to show hidden files when you need to inspect or edit them.

### .replit
The `.replit` file configures how Replit runs the project. It commonly contains a run command and module settings.
Example:

```toml
run = "python main.py"

[nix]
channel = "stable-24_05"
```

Example for a Node development server:
```toml
run = "npm run dev"
```

Example for a Python FastAPI app:
```toml
run = "uvicorn main:app --host 0.0.0.0 --port 8000"
```

Use `.replit` when:
- The Run button starts the wrong command.
- You need a custom development command.
- You need to configure language/runtime behavior.
- You imported a repo Replit could not infer correctly.
- You need deployment-specific build or run commands.
- You need explicit port mapping.

Run commands can be strings:
```toml
run = "npm run dev"
```

or explicit argument arrays:
```toml
run = ["python3", "main.py"]
```

Deployment commands can be separate from development commands:
```toml
[deployment]
build = "npm run build"
run = "npm start"
```

Port mapping example:
```toml
[[ports]]
localPort = 3000
externalPort = 80
```

### replit.nix
Replit uses Nix for system dependencies and reproducible development environments. Use `replit.nix` when the app needs OS-level tools or packages that are not installed through a language package manager.

Example:
```nix
{ pkgs }: {
  deps = [
    pkgs.nodejs_20
    pkgs.postgresql
    pkgs.imagemagick
  ];
}
```

Use language package managers for application libraries:
- `npm` for JavaScript packages.
- `pip`, `uv`, or Poetry for Python packages.
- `cargo` for Rust crates.
- `go mod` for Go modules.

Use Nix for system packages:
- Database clients.
- Image or video tools.
- CLI tools.
- Native libraries required by Python or Node packages.

### replit.md
Replit's docs describe `replit.md` as a project-level guidance file. Agent automatically creates it in the project root for new Agent projects, reads it during future conversations, and can update it as the project evolves. You can also create or edit it manually.

Keep `replit.md` focused. It should contain durable project context, not a giant transcript.

Example:
```markdown
# Project Notes

This is a FastAPI backend with a React frontend.

## Commands

- Backend dev: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Frontend dev: `npm run dev`
- Tests: `pytest && npm test`

## Architecture

- `app/` contains API routes, services, and database models.
- `frontend/` contains the React app.
- Use PostgreSQL for persistent data.

## Rules

- Do not hardcode secrets.
- Keep API response models stable.
- Add tests for auth and billing logic.
```

Use `replit.md` to reduce repeated instructions and prevent Agent from making the same mistakes.

### Package Manifests
Common manifests:
- `package.json` for Node.js.
- `requirements.txt` or `pyproject.toml` for Python.
- `Cargo.toml` for Rust.
- `go.mod` for Go.
- `pom.xml` or `build.gradle` for Java.

If Agent installs dependencies manually but does not update the manifest, future runs or deployments may break. Make sure dependency manifests reflect what the app imports.
### Ports
Web apps must listen on a network interface and port Replit can expose.
General rule:
```text
Host: 0.0.0.0
Port: use the configured or environment-provided port when available
```

Node example:
```javascript
const express = require("express");

const app = express();
const port = process.env.PORT || 3000;

app.get("/", (req, res) => {
  res.json({ status: "ok" });
});

app.listen(port, "0.0.0.0", () => {
  console.log(`listening on ${port}`);
});
```

Python example:
```python
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
```

## Working With Code

### Manual Development
Even when Agent writes code, you should understand how to work manually:
```bash
ls
cat package.json
npm install
npm run dev
npm test
```

For Python:
```bash
python -m pip install -r requirements.txt
python main.py
pytest
```

### Reviewing Agent Changes

After a generation, inspect:

- New files.
- Deleted files.
- Package changes.
- Environment variable assumptions.
- Database schema changes.
- Auth changes.
- Deployment configuration.

Ask:

```text
Summarize exactly what files you changed and why. Include any new commands,
environment variables, database changes, and deployment assumptions.
```

### Refactoring

For large refactors, constrain the blast radius:

```text
Refactor the task filtering logic into a separate module.
Do not change the database schema, routes, or UI behavior.
After the refactor, run the existing tests and fix only regressions caused by the refactor.
```

### Dependency Management

Use standard package files:

Node:

```bash
npm install zod
npm install -D vitest
```

Python:

```bash
python -m pip install pydantic
python -m pip freeze > requirements.txt
```

Prefer explicit manifests over relying on packages that happen to be present in the workspace.

## Secrets and Environment Variables

Secrets are encrypted environment variables. Use them for:

- API keys.
- Database URLs.
- Auth secrets.
- Webhook signing secrets.
- OAuth client secrets.
- Payment provider keys.

Never put secrets in:

- Source code.
- `replit.md`.
- `.replit`.
- Public logs.
- Screenshots.
- Git commits.

### Accessing Secrets

Node:

```javascript
const apiKey = process.env.OPENAI_API_KEY;

if (!apiKey) {
  throw new Error("Missing OPENAI_API_KEY");
}
```

Python:

```python
import os

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")
```

### Secret Naming

Use clear names:

```text
DATABASE_URL
SESSION_SECRET
OPENAI_API_KEY
STRIPE_SECRET_KEY
CLERK_SECRET_KEY
JWT_SECRET
```

Avoid:

```text
KEY
SECRET
TOKEN
PASSWORD
```

### Development vs Deployment Secrets

Confirm whether a secret exists in both the development environment and deployment environment. A common failure is:

1. Preview works.
2. Deployment fails.
3. Logs show missing environment variable.
4. The secret was only configured for one context.

## Databases

Replit provides database integrations, and the official docs describe important changes to development databases after December 4, 2025. In older docs or tutorials you may see different behavior. Treat current Replit docs as the source of truth.

Common database use cases:

- User accounts.
- App settings.
- Tasks, posts, orders, or records.
- Analytics events.
- Durable app state.

### Development Database vs Production Database

Think in two environments:

- **Development database**: used while building and previewing.
- **Production database**: used by a deployed app.

Do not assume data in development automatically appears in production. Do not test production-only migration behavior only in preview.

### DATABASE_URL

Most database-backed apps read a connection string:

```text
DATABASE_URL=postgresql://...
```

Python:

```python
import os

from sqlalchemy import create_engine

database_url = os.environ["DATABASE_URL"]
engine = create_engine(database_url)
```

Node:

```javascript
const postgres = require("postgres");

const sql = postgres(process.env.DATABASE_URL);
```

### Schema Migrations

Use migrations for production data:

```bash
npx prisma migrate dev
npx prisma migrate deploy
```

or:

```bash
alembic revision --autogenerate -m "create tasks"
alembic upgrade head
```

Do not rely on "create tables on startup" for serious production apps. It is acceptable for small demos but weak for real schema evolution.

### Example Prisma Model

```prisma
model User {
  id        String   @id @default(cuid())
  email     String   @unique
  createdAt DateTime @default(now())
  tasks     Task[]
}

model Task {
  id          String   @id @default(cuid())
  title       String
  description String?
  done        Boolean  @default(false)
  ownerId     String
  owner       User     @relation(fields: [ownerId], references: [id])
  createdAt   DateTime @default(now())
}
```

### Database Best Practices

- Keep development and production data separate.
- Use migrations.
- Back up production data.
- Validate ownership in queries.
- Add indexes for common filters.
- Do not log connection strings.
- Store `DATABASE_URL` as a secret.
- Test deployment with production-like data before relying on it.

## App Storage

Replit's object storage product is now referred to as **App Storage** in current docs, though older pages may still mention Object Storage. Use it for files and binary objects that should persist outside the app filesystem.

Use App Storage for:

- User-uploaded images.
- Generated reports.
- Attachments.
- Exported files.
- Media objects.
- Durable blobs.

Do not use App Storage for:

- Relational data that needs joins.
- Secrets.
- Frequently updated tiny state that belongs in a database.
- Files that must be bundled with source code.

### Filesystem vs App Storage

The workspace filesystem is for source code and development. A deployed app filesystem may not be the right place for persistent user uploads. Use App Storage when the object must survive redeploys and scale beyond one process.

### Example Storage Interface

The exact SDK details can change, but the design pattern is stable:

```python
class Storage:
    async def put_bytes(self, key: str, data: bytes, content_type: str) -> str:
        ...

    async def get_bytes(self, key: str) -> bytes:
        ...

    async def delete(self, key: str) -> None:
        ...
```

Keep storage access behind a small module so you can adapt if SDK calls or providers change.

### Storage Key Design

Use predictable but non-sensitive keys:

```text
users/{user_id}/avatars/{avatar_id}.png
reports/{report_id}/export.csv
attachments/{task_id}/{attachment_id}
```

Avoid putting secrets or raw emails in object keys.

### Upload Flow

Typical upload flow:

1. Validate authentication.
2. Validate file size and content type.
3. Generate a storage key.
4. Upload to App Storage.
5. Store metadata in the database.
6. Return the file URL or object ID.

Metadata table example:

```sql
create table files (
  id text primary key,
  owner_id text not null,
  storage_key text not null unique,
  filename text not null,
  content_type text not null,
  size_bytes integer not null,
  created_at timestamp not null default current_timestamp
);
```

## Authentication

Replit docs include authentication options and current references to Clerk Auth. The right choice depends on the app.

Common choices:

- Replit Auth for simple Replit-user identity flows.
- Clerk for production-ready user management.
- A custom auth system using sessions or JWTs.
- OAuth with an external identity provider.

### Choosing Auth

Use a managed provider when:

- Users need password reset.
- You need social login.
- You need MFA.
- You need organization/team auth.
- You do not want to own credential security.

Use custom auth only when:

- The requirements are simple and well understood.
- You have a reason not to use a managed provider.
- You can handle password hashing, sessions, CSRF, and account recovery correctly.

### Auth Questions to Answer

Before building:

```text
1. Who can sign up?
2. Do users log in with email, social auth, or Replit identity?
3. Are there roles?
4. Can users belong to organizations?
5. What data does each user own?
6. What happens when a user is deleted?
7. Are sessions cookie-based or token-based?
8. Does the deployed app need private access?
```

### Authorization Pattern

Always enforce ownership server-side:

```javascript
app.get("/api/tasks/:id", requireUser, async (req, res) => {
  const task = await db.task.findFirst({
    where: {
      id: req.params.id,
      ownerId: req.user.id
    }
  });

  if (!task) {
    res.status(404).json({ error: "Task not found" });
    return;
  }

  res.json(task);
});
```

Do not fetch by ID and then rely only on frontend hiding.

## Previewing and Debugging

### Development Preview

Preview shows the currently running app. It is not the same as a production deployment.

Use preview to verify:

- The app starts.
- Routes render.
- Forms submit.
- Data persists.
- Auth flow works.
- Errors are visible in logs.

### Common Debug Loop

1. Reproduce the problem.
2. Read Console output.
3. Run the command manually in Shell.
4. Identify whether it is a build, runtime, environment, database, or UI issue.
5. Ask Agent for a minimal fix.
6. Verify by rerunning the app or tests.

Prompt:

```text
Do not rewrite the app. Diagnose this error from the console, explain the root
cause, then make the smallest code/config change needed. Afterward, run the
relevant command to verify.
```

### Build Errors

Build errors usually come from:

- Missing dependency.
- Wrong package version.
- Syntax error.
- Type error.
- Missing generated client.
- Misconfigured build script.

Commands:

```bash
npm run build
npm test
python -m pytest
python -m compileall .
```

### Runtime Errors

Runtime errors usually come from:

- Missing environment variable.
- Wrong port.
- Database connection failure.
- Invalid route handler.
- Auth configuration issue.
- File path assumption.

### UI Bugs

For UI bugs, give Agent concrete reproduction steps:

```text
On the dashboard, click "New Task", leave title empty, then submit.
Expected: inline validation error.
Actual: modal closes and creates a blank task.
Fix only this workflow.
```

## Publishing and Deployments

Publishing turns a working artifact into a deployed app. Replit docs describe several deployment types. Choose based on runtime needs.

### Static Deployments

Use static deployments for:

- HTML/CSS/JS sites.
- Vite or React builds.
- Documentation sites.
- Marketing pages.
- Apps that call external APIs but do not run a backend server on Replit.

Typical build output:

```text
dist/
build/
public/
```

Example:

```bash
npm run build
```

Deployment needs:

- Build command.
- Public directory.
- Environment variables if the frontend build uses them.

### Autoscale Deployments

Use autoscale deployments for web services that should scale with traffic:

- APIs.
- Full-stack web apps.
- Backends.
- Apps with variable request load.

Typical requirements:

- A start command.
- Correct port binding.
- Production environment variables.
- Database connection.
- Health endpoint.

Example start command:

```bash
npm run start
```

or:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Reserved VM Deployments

Use reserved VM deployments when the app needs dedicated compute resources or more predictable always-running behavior.

Useful for:

- Stateful-ish services that need a consistent runtime.
- Long-running web apps.
- Apps with steady background work.
- Workloads that are not a fit for request-based autoscaling.

Still design for restarts. A reserved VM is not a substitute for durable storage, backups, or process supervision.

### Scheduled Deployments

Use scheduled deployments for jobs that run on a schedule:

- Daily reports.
- Cleanup jobs.
- Data imports.
- Periodic notifications.
- Sync tasks.

Example job shape:

```python
def main() -> None:
    print("running scheduled job")
    # fetch data, update database, send notifications


if __name__ == "__main__":
    main()
```

Scheduled jobs should be idempotent when possible. If the job runs twice, it should not corrupt data.

### Private Deployments

Private deployments restrict access to your app. Use them for:

- Internal tools.
- Admin panels.
- Client demos.
- Apps still under review.

Do not rely on obscurity alone. If the app contains sensitive data, enforce authentication and authorization inside the app too.

### Deployment Checklist

Before deployment:

- Preview works.
- Build command succeeds.
- Start command succeeds.
- Correct deployment type selected.
- App binds to `0.0.0.0`.
- App uses `PORT` when required.
- Production secrets are present.
- Database is production-ready.
- Migrations have run.
- Auth callback URLs match deployed domain.
- Uploaded files use App Storage, not local temp files.
- Logs are readable.
- Health endpoint exists for server apps.

After deployment:

- Open the public URL.
- Test signup/login.
- Test the main workflow.
- Check logs.
- Confirm data persistence.
- Confirm environment-specific URLs.
- Redeploy after code changes.

## Domains, Sharing, and Access Control

### Development Sharing

During development, you can share a project or preview with collaborators depending on workspace settings. Be careful with:

- Secrets visible in logs.
- Admin routes.
- Unfinished auth.
- Test data.
- Generated code you have not reviewed.

### Custom Domains

For production apps, a custom domain usually requires:

1. Deploy the app.
2. Add the domain in Replit deployment settings.
3. Configure DNS records at your domain registrar.
4. Wait for propagation.
5. Verify HTTPS.
6. Update auth callback URLs and CORS origins.

Common DNS records:

```text
CNAME www -> target provided by Replit
A or ALIAS apex -> target provided by Replit or DNS provider
```

Use the exact DNS values Replit gives you.

### CORS and Domains

If your frontend and backend are separate artifacts or domains, configure CORS explicitly.

Example:

```javascript
app.use(cors({
  origin: ["https://app.example.com"],
  credentials: true
}));
```

Avoid wildcard CORS for authenticated apps.

## Complete Project Walkthrough

This walkthrough builds a small internal issue tracker using Replit Agent plus manual verification.

### Goal

Build an issue tracker with:

- Login.
- Projects.
- Issues.
- Statuses: open, in progress, blocked, done.
- Priority.
- Assignee.
- Comments.
- Dashboard.
- Database persistence.
- Deployment.

### Step 1: Start With Plan Mode

Prompt:

```text
Build an internal issue tracker for a small software team.

Required workflows:
- Users can sign up and log in.
- Users can create projects.
- Users can create issues inside projects.
- Issues have title, description, status, priority, assignee, and due date.
- Users can comment on issues.
- Dashboard shows open issues, blocked issues, and overdue issues.

Use a real database. Use a clean, dense, work-focused UI.
Before building, propose:
1. Stack.
2. Data model.
3. Main screens.
4. API routes.
5. Deployment type.
6. Risks and assumptions.

Wait for approval before building.
```

Review the plan. Push back on vague parts:

```text
Tighten the plan:
- Use PostgreSQL-compatible storage.
- Add server-side authorization for project membership.
- Include empty states and validation errors.
- Add tests for issue ownership and status updates.
```

### Step 2: Approve a Specific Build

Prompt:

```text
Proceed with the build.

Keep the first version scoped:
- Auth.
- Project CRUD.
- Issue CRUD.
- Comment creation.
- Dashboard counts.
- Basic responsive UI.

Do not add notifications, billing, file uploads, or AI features yet.
Create a checkpoint after the first working version.
```

### Step 3: Inspect the Generated Structure

Ask:

```text
Summarize the project structure. Explain:
- The app entry point.
- The database schema.
- The auth flow.
- The main routes.
- The commands to run, test, and deploy.
```

Then inspect manually:

```bash
ls
cat package.json
cat .replit
find . -maxdepth 2 -type f | sort
```

### Step 4: Verify Preview

Test:

```text
1. Create account.
2. Create project.
3. Create issue.
4. Change status.
5. Add comment.
6. Reload.
7. Confirm data persists.
8. Try viewing another user's issue if possible.
```

### Step 5: Add Tests

Prompt:

```text
Add focused tests for:
- Creating an issue requires login.
- Users can only read issues in projects they belong to.
- Status can only be one of open, in_progress, blocked, done.
- Dashboard counts match issue data.

Run the tests and fix failures.
```

### Step 6: Production Readiness Review

Prompt:

```text
Review the app for deployment readiness.

Check secrets, database, migrations, auth callback URLs, CORS,
start command, build command, port binding, logs, and health checks.

Return a checklist first. Do not make changes until I approve.
```

### Step 7: Deploy

Choose deployment type:

- Static if it is only a frontend.
- Autoscale for normal web app/API.
- Reserved VM if it needs dedicated long-running compute.
- Scheduled if it is a periodic job.
- Private if access should be restricted.

Deploy, then test the deployed URL separately from preview.

### Step 8: Maintain

For future features:

```text
Add issue labels.
Before editing, explain the schema change, UI changes, and migration plan.
Keep the existing issue workflow working.
```

This prevents a simple feature from becoming an accidental rewrite.

## Testing and Quality

### What to Test

Test critical behavior:

- Auth.
- Authorization.
- Database writes.
- Form validation.
- Main workflows.
- Payment or external API integration, if any.
- Deployment start command.
- Build command.

### JavaScript Tests

Example:

```javascript
import { describe, expect, it } from "vitest";

function normalizeStatus(status) {
  const allowed = ["open", "in_progress", "blocked", "done"];
  if (!allowed.includes(status)) {
    throw new Error("invalid status");
  }
  return status;
}

describe("normalizeStatus", () => {
  it("accepts a valid status", () => {
    expect(normalizeStatus("blocked")).toBe("blocked");
  });

  it("rejects invalid status", () => {
    expect(() => normalizeStatus("bad")).toThrow("invalid status");
  });
});
```

### Python Tests

Example:

```python
import pytest


def validate_priority(priority: str) -> str:
    allowed = {"low", "medium", "high"}
    if priority not in allowed:
        raise ValueError("invalid priority")
    return priority


def test_validate_priority_accepts_valid_value() -> None:
    assert validate_priority("high") == "high"


def test_validate_priority_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        validate_priority("urgent")
```

### Manual QA Checklist

- App starts from the Run button.
- App starts from the deployment start command.
- Preview renders.
- Main navigation works.
- Forms validate input.
- Data persists after reload.
- Errors are understandable.
- Auth protects private pages.
- Logs do not leak secrets.
- Mobile layout works.
- Deployment URL works.

### Asking Agent to Test

Prompt:

```text
Run the test suite and the production build command.
If either fails, explain the root cause and make the smallest fix.
After fixing, run the commands again and summarize results.
```

## Best Practices

### Using Agent

- Start with Plan Mode for meaningful apps.
- Require approval before large builds.
- Create checkpoints at working milestones.
- Ask for summaries of changed files.
- Keep prompts scoped.
- Paste exact errors.
- Ask Agent to verify with commands.
- Do not accept broad rewrites unless you intended one.
- Use stronger Agent modes for harder tasks, not every small edit.

### Product Design

- Define the target user.
- Define the core workflow.
- Keep the first version small.
- Add persistence early if data matters.
- Add auth before adding sensitive data.
- Test the deployed URL, not only preview.
- Avoid building billing, AI, file uploads, and notifications all at once.

### Code Quality

- Keep dependencies in manifests.
- Keep start/build/test commands documented.
- Use `replit.md` for project-specific guidance.
- Separate frontend, backend, and database code when the app grows.
- Keep business logic outside UI components where possible.
- Add tests around critical rules.
- Prefer migrations for schema changes.

### Security

- Store secrets in Secrets.
- Never hardcode API keys.
- Enforce server-side authorization.
- Use managed auth for serious apps.
- Validate all input on the server.
- Keep CORS restrictive.
- Do not log tokens or connection strings.
- Rotate secrets if they appear in logs or code.
- Use private deployments for internal tools when appropriate.

### Deployment

- Pick deployment type based on runtime behavior.
- Confirm start command.
- Confirm build command.
- Bind to `0.0.0.0`.
- Use the expected port.
- Configure production secrets.
- Run migrations intentionally.
- Use App Storage for persistent files.
- Add health checks for server apps.
- Read deployment logs after every release.

### Data

- Treat preview and production data separately.
- Back up important production data.
- Use App Storage for blobs and a database for relational state.
- Store metadata for uploaded files in the database.
- Do not depend on local files for user uploads in deployed apps.
- Test data ownership boundaries.

### Collaboration

- Use comments and project guidance files.
- Document commands.
- Keep generated code understandable.
- Review before sharing or publishing.
- Avoid exposing unfinished admin features.

## Troubleshooting

### Preview Is Blank

Check:

- Is the app running?
- Did the server bind to `0.0.0.0`?
- Is it using the correct port?
- Did the frontend build fail?
- Are there browser console errors?
- Are required environment variables missing?

Prompt:

```text
The preview is blank. Inspect the console and start command.
Find whether this is a frontend build issue, server startup issue, or port issue.
Make the smallest fix and verify preview loads.
```

### Run Button Starts the Wrong Command

Fix `.replit`:

```toml
run = "npm run dev"
```

or:

```toml
run = "python main.py"
```

Then rerun.

### Missing Secret

Symptoms:

- `Missing DATABASE_URL`.
- `Missing API key`.
- Auth provider initialization fails.
- Deployment works in preview but fails after publishing.

Fix:

- Add the secret in the correct environment.
- Confirm the variable name exactly matches code.
- Restart the app or redeploy.

### Database Works in Preview but Not Deployment

Check:

- Production `DATABASE_URL`.
- Database permissions.
- Migration status.
- SSL requirements.
- Whether the app is reading the right environment variable.
- Whether preview and production databases are different.

### Deployment Fails

Check:

- Build command.
- Start command.
- Port binding.
- Missing secrets.
- Package manifest.
- Database migrations.
- Runtime version.
- Logs.

Prompt:

```text
Deployment failed. Use the deployment logs to identify the first real error.
Do not guess from symptoms. Fix the underlying build/start/config problem and
explain what changed.
```

### Agent Broke the App

Use checkpoints:

1. Identify last working checkpoint.
2. Compare what changed.
3. Revert if the change is too broad.
4. Re-ask with a smaller prompt.

Prompt:

```text
The last generation broke the app. Compare the broken state to the last working
checkpoint. Explain the likely cause. Do not rewrite; either revert the bad
change or make a minimal fix.
```

### App Loses Uploaded Files

Likely cause: storing user uploads on local filesystem instead of App Storage.

Fix:

- Move uploads to App Storage.
- Store metadata in database.
- Avoid assuming local files persist across deployments.

### Auth Redirects to the Wrong URL

Check:

- Deployment URL.
- Custom domain.
- Auth provider callback URL.
- CORS origin.
- Cookie domain.
- HTTPS settings.

### Package Installed but Import Fails

Check:

- Did the package get added to `package.json`, `requirements.txt`, or `pyproject.toml`?
- Was the install command run in the right directory?
- Is the runtime using the same environment?
- Does the deployment install dependencies from the manifest?

## Quick Reference

### New App Prompt

```text
Build [app] for [users]. It should support [workflows].
Use [stack] if reasonable. Use persistent storage.
Start in Plan Mode with architecture, data model, screens, routes,
deployment type, and risks. Wait for approval before building.
```

### Debug Prompt

```text
Here is the exact error: [paste error].
Reproduce it, identify the root cause, make the smallest fix, and verify with
the relevant command.
```

### Production Readiness Prompt

```text
Review this app for production readiness: secrets, database, auth, migrations,
CORS, start command, build command, port binding, logs, health checks, and
deployment type. Return a checklist before making changes.
```

### Common Commands

```bash
npm install
npm run dev
npm run build
npm test
python -m pip install -r requirements.txt
python main.py
pytest
```

### Common Config

```toml
run = "npm run dev"
```

```toml
run = "uvicorn main:app --host 0.0.0.0 --port 8000"
```

### Common Secrets

```text
DATABASE_URL
SESSION_SECRET
JWT_SECRET
CLERK_SECRET_KEY
OPENAI_API_KEY
STRIPE_SECRET_KEY
```

### Deployment Selection

```text
Static: frontend-only sites.
Autoscale: normal web apps and APIs.
Reserved VM: dedicated always-running compute.
Scheduled: recurring jobs.
Private: restricted access apps.
```

## Sources

Primary official docs consulted:

- Replit documentation home: https://docs.replit.com/
- Projects reference: https://docs.replit.com/references/projects-and-artifacts/projects
- Artifacts reference: https://docs.replit.com/references/projects-and-artifacts/artifacts
- Replit Agent overview: https://docs.replit.com/references/agent/overview
- Agent modes: https://docs.replit.com/references/agent/agent-modes
- Plan Mode: https://docs.replit.com/references/agent/plan-mode
- Task system: https://docs.replit.com/references/agent/task-system
- Replit Assistant: https://docs.replit.com/core-concepts/ai/replit-assistant
- App configuration, `.replit`, and `replit.nix`: https://docs.replit.com/replit-app/configuration
- `replit.md`: https://docs.replit.com/references/project-setup/replit-dot-md
- Secrets: https://docs.replit.com/core-concepts/project-editor/app-setup/secrets
- Ports: https://docs.replit.com/references/project-setup/ports
- Dependency management: https://docs.replit.com/references/project-setup/dependency-management
- SQL database: https://docs.replit.com/references/data-and-storage/sql-database
- Production databases: https://docs.replit.com/references/data-and-storage/production-databases
- App Storage / Object Storage: https://docs.replit.com/references/data-and-storage/object-storage
- Authentication: https://docs.replit.com/references/auth-and-identity/authentication
- Clerk Auth: https://docs.replit.com/references/auth-and-identity/clerk-auth
- Publishing and deployments overview: https://docs.replit.com/learn/projects-and-artifacts/replit-deployments
- Static deployments: https://docs.replit.com/references/publishing/static-deployments
- Autoscale deployments: https://docs.replit.com/references/publishing/autoscale-deployments
- Reserved VM deployments: https://docs.replit.com/references/publishing/reserved-vm-deployments
- Scheduled deployments: https://docs.replit.com/references/publishing/scheduled-deployments
- Private deployments: https://docs.replit.com/references/publishing/private-deployments
- Custom domains: https://docs.replit.com/references/publishing/custom-domains
