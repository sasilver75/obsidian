
# TypeScript the Language

TypeScript (TS) is a statically-typed superset of JavaScript developed by Microsoft (2012, Anders Hejlsberg — the same person behind Turbo Pascal, Delphi, and C#). It compiles to plain JS and runs anywhere JS runs: browsers, [[Node.js]], [[Deno]], [[Bun]], [[Cloudflare Workers]], etc.

**Mental model from Python/Go:** TypeScript is JavaScript with a ==structural type system bolted on at the compiler level==. The types are erased at build time — ==at runtime, it's just JavaScript==. This is the single most important thing to internalize. Your `interface User { id: number }` does not exist when the program runs. You cannot do `if (x instanceof User)`. The compiler is a linter with a very fancy type checker; the runtime is unchanged.

It is now ==the default language of the web frontend== (React, Vue, Svelte, Solid all assume TS) and increasingly the default for the backend ([[Next.js]], [[Hono]], [[tRPC]], [[Bun]]'s ecosystem).


## Key Design Decisions (the "why is this weird" section)

### ==Types are erased at compile time==
```ts
// What you write:
interface User { id: number; name: string }
function greet(u: User) { return `Hello ${u.name}` }

// What runs (after tsc):
function greet(u) { return `Hello ${u.name}` }
```
Implications:
- ==No runtime type checks==. Pass a `{ id: "abc", name: 5 }` from an API and TS won't catch it — you need [[Zod]]/[[Valibot]]/[[ArkType]] at the boundary.
- No reflection on types. Libraries like `class-validator` rely on decorators or codegen to get type info at runtime.

### Structural typing
TS doesn't care about names — it cares about ==shape==. Like Go interfaces, but more aggressive: two unrelated types with the same fields are interchangeable.

```ts
interface Point2D { x: number; y: number }
interface Vec2     { x: number; y: number }

const p: Point2D = { x: 1, y: 2 }
const v: Vec2 = p  // fine — same shape
```

### `any`, `unknown`, `never` — the type-system escape hatches
- `any`: ==turns off the type checker for this value==. Avoid. The compiler stops helping you.
- `unknown`: "I don't know the type — you must narrow it before use." The ==safe version of `any`==.
- `never`: a value that ==cannot exist== (e.g., the return type of `throw`). Used for exhaustiveness checks.

```ts
function assertNever(x: never): never { throw new Error(`unexpected: ${x}`) }

switch (kind) {
  case "a": return ...
  case "b": return ...
  default:  return assertNever(kind)  // compile error if a new variant is added
}
```

### Union and discriminated union types
TS's killer feature for modeling domain state.
```ts
type Result<T> =
  | { ok: true;  value: T }
  | { ok: false; error: string }

function handle(r: Result<number>) {
  if (r.ok) console.log(r.value)   // TS knows .value exists here
  else      console.log(r.error)   // and .error here
}
```
==This is how you model errors in idiomatic TS== — closer to Go/Rust than to thrown exceptions.

### Generics
First-class, very expressive. Often combined with constraints (`extends`) and conditional types.
```ts
function first<T>(arr: T[]): T | undefined { return arr[0] }

type ApiResponse<T> = { data: T; status: number }
```

### ==Conditional and mapped types== — the "type-level programming" layer
TS's type system is genuinely [[Turing-Complete]]. You'll see things like:
```ts
type Partial<T>  = { [K in keyof T]?: T[K] }
type Pick<T, K extends keyof T> = { [P in K]: T[P] }
type ReturnType<F> = F extends (...args: any) => infer R ? R : never
```
You don't have to write these — but understanding the standard `Partial`, `Pick`, `Omit`, `Record`, `ReturnType`, `Awaited`, etc. is core literacy.

### `strict` mode is non-negotiable
Set `"strict": true` in `tsconfig.json`. Without it, you have ~half a type system. ==Treat non-strict TS as legacy.==

### Async/await everywhere
Just like JS. ==Promise-based==, single-threaded event loop. No goroutines, no threads (use Worker threads or processes for true parallelism).

```ts
async function fetchUser(id: number): Promise<User> {
  const res = await fetch(`/api/users/${id}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<User>   // ⚠ no validation — type assertion is a lie
}
```

### Errors are still `throw`/`catch` (but mostly typed as `unknown`)
TS 4.4+ makes `catch (e)` default to `unknown`. ==You must narrow before use==.
```ts
try { ... }
catch (e) {
  if (e instanceof Error) console.error(e.message)
  else                    console.error("unknown error", e)
}
```
Increasingly, idiomatic TS uses Result-like types ([[neverthrow]], or hand-rolled discriminated unions) at module boundaries.

---

# Runtimes

This is the big shift of the last few years — ==TS is no longer just "JS for browsers."==

- ==[[Node.js]]==: The incumbent. v22 LTS as of 2025. Has built-in `--experimental-strip-types` since v22.6 — can run `.ts` files directly without compilation for many cases.
- ==[[Bun]]==: All-in-one runtime + bundler + package manager + test runner. ==Runs TS natively, no compile step==. Drop-in Node-compatible for most APIs. Genuinely fast.
- ==[[Deno]]==: Ryan Dahl's "Node redo." TS-native, secure-by-default (permissions model), Web-API-first, has a built-in fmt/lint/test. v2 (2024) restored npm compatibility.
- ==Edge runtimes==: [[Cloudflare Workers]], [[Vercel]] Edge, Deno Deploy — V8 isolates, no Node APIs by default, Web-standards-first (`fetch`, `Request`, `Response`).
- Browsers: still the original target; TS compiles down to whatever ES version you specify.

Rule of thumb (2025-ish):
- New backend service: ==Bun or Node 22==. Bun if you value speed & simplicity; Node if you value ecosystem maturity & corporate trust.
- Edge/global low-latency: Cloudflare Workers + Hono.
- New CLI tool: Bun (single-binary builds via `bun build --compile`).


# Toolchain

The JS toolchain ==used to be a nightmare==. It's converging fast.

### Package managers
- ==`npm`==: Default with Node. Slow installs, but universal.
- ==`pnpm`==: Hard-linked content-addressable store. Fast, disk-efficient, ==the default for new projects==. Native monorepo workspaces.
- ==`yarn`==: Berry (v4+) is solid; less momentum than pnpm now.
- ==`bun`==: Bun's own package manager. Fastest installer by a wide margin. Reads `package.json`, writes `bun.lock`.

### Compilers / type checkers
- ==`tsc`==: The official TypeScript compiler. ==Slow but authoritative==. You'll always use it for type-checking in CI even if you don't use it for transpiling.
- ==`swc`==: Rust-based JS/TS transpiler. Used by [[Next.js]] internally. ==~20× faster== than tsc for transpilation — but doesn't type-check.
- ==`esbuild`==: Go-based bundler/transpiler. Insanely fast. Same trade-off: transpile-only, no type-check.
- ==`tsx`==: Wrapper for running `.ts` files directly in dev (via esbuild). Replaces `ts-node` for most uses.

### Bundlers (browser/edge)
- ==[[Vite]]==: ==The default for new frontend projects==. esbuild for dev, Rollup for prod builds, plugin ecosystem.
- [[Turbopack]]: Vercel's Rust bundler, default in [[Next.js]] dev mode.
- ==[[esbuild]] / [[Rollup]] / [[Webpack]]==: Webpack is legacy; Rollup is for libraries; esbuild for speed.
- ==Rolldown==: Rust-based Rollup-compatible bundler. Under active dev; expected to subsume Vite's prod path.

### Linters / formatters
- ==[[ESLint]]==: The standard linter. v9+ uses flat config (`eslint.config.js`).
- ==[[Prettier]]==: Opinionated formatter. ==Run it. Don't argue about style==.
- ==[[Biome]]==: Rust-based all-in-one linter + formatter, ESLint+Prettier replacement. ==Rapidly gaining ground== for new projects; ~25× faster.
- ==`oxlint`==: Even faster Rust linter (oxc). Newer; complements/replaces ESLint.

### Why so many Rust tools?
The bottleneck for JS/TS dev is the toolchain itself. The whole ecosystem is being rewritten in Rust/Go for ==1-2 orders of magnitude speed-ups==: Turbopack, Biome, oxc, swc, rolldown, etc.


# `tsconfig.json` — the one config that matters

A reasonable modern baseline:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "verbatimModuleSyntax": true,
    "declaration": true,
    "outDir": "./dist"
  }
}
```
Notable knobs:
- `noUncheckedIndexedAccess`: `arr[0]` becomes `T | undefined`. Catches a whole class of bugs.
- `isolatedModules`: ensures each file can be transpiled in isolation (required by swc/esbuild/Bun).
- `verbatimModuleSyntax`: forces explicit `import type { ... }` for type-only imports — required for clean ESM interop.


# Project Layout

No enforced layout, but common patterns:

```
my-app/
├── src/
│   ├── index.ts            # entry
│   ├── lib/                # internal modules
│   ├── routes/             # http handlers (api)
│   └── types/              # shared types
├── tests/
├── package.json
├── tsconfig.json
├── eslint.config.js
└── pnpm-lock.yaml
```

In a monorepo (very common), you'll see:
```
repo/
├── apps/
│   ├── web/                # Next.js / Vite app
│   └── api/                # Hono / Express backend
├── packages/
│   ├── ui/                 # shared React components
│   ├── db/                 # Drizzle / Prisma schema + client
│   └── config/             # shared eslint/tsconfig
├── pnpm-workspace.yaml
└── turbo.json              # Turborepo task pipeline
```
==[[Turborepo]]== or [[Nx]] orchestrate task pipelines across packages.


# Web Frameworks (Frontend)

- ==[[React]]==: Still dominant. Pair with TS via [[Vite]] + [[React Router]] / [[TanStack Router]] for SPAs.
- ==[[Next.js]]==: The full-stack React framework. App Router (RSC), Server Actions, file-based routing. ==Default choice for production React apps.==
- ==[[Remix]] / [[React Router v7]]==: Web-standards-focused React framework; merged into React Router.
- ==[[Astro]]==: Content-heavy sites (docs, blogs, marketing). "Islands" architecture, ships zero JS by default.
- ==[[SvelteKit]]== / ==[[SolidStart]]== / ==[[Qwik]]==: Non-React alternatives, all TS-first.
- ==[[TanStack Start]]==: New full-stack TS framework from the TanStack team (Query, Router, Table). Up-and-coming.


# Web Frameworks (Backend)

In rough order of "what new TS services pick in 2025":

- ==[[Hono]]==: Lightweight, ==web-standards-based== (`Request`/`Response`), runs on Node/Bun/Deno/Workers/Vercel Edge. ==The clean default for new APIs==. Great types, RPC mode.
- ==[[Express]]==: Still ubiquitous, still works. Types are bolted on (`@types/express`). Use it for legacy/familiarity.
- ==[[Fastify]]==: Schema-driven, very fast, good types. The mature "Express but better" choice.
- ==[[tRPC]]==: ==End-to-end typesafe APIs without codegen==. Backend defines procedures, frontend imports them as typed functions. Killer DX in TS monorepos. Note: only works TS↔TS.
- ==[[Elysia]]==: Bun-first framework, big on type-level performance.
- ==[[NestJS]]==: Heavy, Angular-flavored, decorators + DI. Used in enterprise. Less common in greenfield.


# Database

The TS DB story has gotten ==very good== recently.

- ==[[Drizzle ORM]]==: ==The current default for new projects==. SQL-first, lightweight, generates types from your schema, no codegen step needed at dev time. Works with Postgres, MySQL, SQLite, [[Cloudflare D1]], [[Turso]], [[Neon]], etc.
  ```ts
  const users = pgTable('users', { id: serial('id').primaryKey(), name: text('name') })
  const u = await db.select().from(users).where(eq(users.id, 1))  // fully typed
  ```
- ==[[Prisma]]==: The incumbent. Schema-first (Prisma schema DSL → codegen → typed client). Great DX, but a heavier runtime and historically slow query engine (now being rewritten in Rust). Still very common in production.
- ==[[Kysely]]==: Type-safe SQL query builder. No ORM abstraction — you write SQL-shaped TS. The choice if you want zero magic.
- ==`postgres`== (porsager/postgres) / ==`pg`==: Raw drivers. `postgres` is the fast modern choice.
- ==[[Supabase]]==: Postgres-as-a-service with a typed TS client generated from your schema.

### Migrations
- ==Drizzle Kit==: Bundled with Drizzle. `drizzle-kit generate` + `drizzle-kit migrate`.
- ==Prisma Migrate==: Bundled with Prisma.
- ==`node-pg-migrate`== / `dbmate`: Framework-agnostic, plain SQL.


# Validation / Runtime Types

Since TS types are erased, you need runtime schemas at boundaries (HTTP, env vars, DB ↔ external systems, LLM outputs, etc.).

- ==[[Zod]]==: ==The de-facto standard.== Define schema once, get TS type + runtime validation.
  ```ts
  const User = z.object({ id: z.number(), name: z.string() })
  type User = z.infer<typeof User>
  const u = User.parse(jsonFromApi)  // throws if invalid
  ```
- ==[[Valibot]]==: Same idea, tree-shakable, ~10× smaller bundle. Great for the edge/frontend.
- ==[[ArkType]]==: Faster, more TS-native syntax (`type({ name: "string" })`).
- ==[[Effect Schema]]==: Part of the [[Effect]] ecosystem; very powerful, steeper learning curve.

Rule of thumb: ==Zod for most things, Valibot when bundle size matters==.


# Async, Concurrency, and the Event Loop

TS/JS is ==single-threaded, event-loop driven==. There are no threads in user code. Concurrency = `Promise.all` / `Promise.allSettled` / `Promise.race`.

```ts
const [users, orders] = await Promise.all([
  fetchUsers(),
  fetchOrders(),
])
```

For ==true parallelism==:
- `Worker` (Web Workers in the browser, `worker_threads` in Node) — message-passing, separate V8 isolates.
- `child_process` / `Bun.spawn` for shelling out.

For ==structured concurrency / cancellation==:
- `AbortController` + `AbortSignal` — the standard cancellation primitive.
  ```ts
  const ctrl = new AbortController()
  fetch(url, { signal: ctrl.signal })
  setTimeout(() => ctrl.abort(), 5000)
  ```
- [[Effect]] / [[Neverthrow]] for higher-level fp-style error/concurrency handling (niche but growing).


# Testing

- ==[[Vitest]]==: ==The current default==. Vite-native, Jest-compatible API, fast, native ESM + TS support, watch mode is excellent.
- ==[[Jest]]==: The legacy incumbent. Still common, slower, ESM story is awkward.
- ==`node:test`== (stdlib, Node 20+): Built-in test runner. Good for libraries that don't want a dep.
- ==`bun test`==: Bun's built-in test runner. Jest-compatible API, very fast.
- ==[[Playwright]]==: ==The default for browser E2E==. Multi-browser, autowait, trace viewer. Replaces [[Cypress]] for most new projects.
- ==[[Testcontainers]]== (`testcontainers-node`): Spin up real DBs/services in Docker for integration tests.
- ==`msw`== (Mock Service Worker): Intercept fetch at the network level for testing.

```ts
// vitest example
import { describe, it, expect } from 'vitest'

describe('add', () => {
  it('adds two numbers', () => {
    expect(add(1, 2)).toBe(3)
  })
})
```


# Logging & Observability

- ==[[Pino]]==: Fast structured JSON logger for Node/Bun. The current default.
- `console.log`: Still fine for small services / scripts.
- ==[[OpenTelemetry]] JS SDK== (`@opentelemetry/*`): Vendor-neutral traces + metrics + logs.
- ==[[Sentry]]==: Error reporting, plus performance & traces. Ubiquitous in product apps.
- For Workers/Edge: structured `console.log` + the platform's ingestion (Cloudflare Workers Logs, Vercel Logs).


# HTTP Client

- ==`fetch`== (stdlib in modern Node, Bun, Deno, browsers): ==Use it.== No library needed for most cases.
- ==[[Axios]]==: Still common. Better defaults (auto-JSON, interceptors). Heavy.
- ==`ky`==: Tiny `fetch` wrapper with retries, hooks, timeouts. The middle ground.
- ==[[ofetch]]==: Nuxt's `fetch` wrapper, nice defaults.

Set timeouts explicitly — `fetch` doesn't time out by default:
```ts
const res = await fetch(url, { signal: AbortSignal.timeout(10_000) })
```


# Monorepo Tools

- ==[[Turborepo]]==: ==The current default==. Task graph caching, parallel execution, remote cache. Vercel-owned.
- ==[[Nx]]==: Heavier, more opinionated, plugin-rich. Common in enterprise / Angular shops.
- ==pnpm workspaces==: Just package management; pair with Turbo for task running.
- ==[[Moon]]==: Newer Rust-based alternative.


# Common Patterns and Idioms

### `as const` for literal narrowing
```ts
const ROLES = ['admin', 'user', 'guest'] as const
type Role = typeof ROLES[number]   // 'admin' | 'user' | 'guest'
```

### Type-only imports
```ts
import type { User } from './types'   // erased at build time, zero runtime cost
```

### Branded types for nominal typing
```ts
type UserId = number & { readonly __brand: 'UserId' }
function asUserId(n: number): UserId { return n as UserId }
// Now you can't accidentally pass a generic number where a UserId is expected
```

### Exhaustive switches with `never`
```ts
type Shape = { kind: 'circle'; r: number } | { kind: 'square'; side: number }
function area(s: Shape): number {
  switch (s.kind) {
    case 'circle': return Math.PI * s.r ** 2
    case 'square': return s.side ** 2
    default: { const _: never = s; throw new Error(`bad shape: ${_}`) }
  }
}
```

### Result-style error handling at boundaries
```ts
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E }
```

### ==Parse, don't validate==
Run user input through `z.parse(...)` at the boundary, get a typed object, ==trust it everywhere downstream==. Don't re-check fields in business logic.


# Deployment

### Single binaries
- ==`bun build --compile`==: Compiles to a single self-contained executable. ==Killer feature.==
- `pkg` / `nexe`: Node equivalents, older.

### Containers
Standard pattern:
```dockerfile
FROM oven/bun:1 AS builder
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile
COPY . .
RUN bun build ./src/index.ts --target=bun --outdir=./dist

FROM oven/bun:1
COPY --from=builder /app/dist /app
CMD ["bun", "/app/index.js"]
```

### Serverless / edge
- [[Vercel]] / [[Netlify]]: deploy from git, zero-config for Next.js.
- ==[[Cloudflare Workers]]==: Wrangler CLI; ==V8 isolates, no Node APIs by default==. Use Hono.
- AWS Lambda: still common, esp. via SST / Serverless Framework / CDK.


# What's in "the stdlib"

JS/TS has a smaller standard library than Python/Go — but it grew a lot recently:
- `fetch`, `Request`, `Response`, `Headers`, `URL`, `URLSearchParams` (Web Fetch API)
- `AbortController` / `AbortSignal`
- `structuredClone` (deep clone)
- `Intl.*` (locale-aware formatting)
- `crypto.subtle` (Web Crypto API)
- `fs/promises`, `path`, `os`, `process` (Node-only)
- `node:test`, `node:assert` (Node 20+)


# Trends

- ==Rust/Go-based tooling has won==: Biome, oxc, swc, Turbopack, Rolldown, esbuild. ESLint and tsc are the last big JS-based tools standing.
- ==Bun is real==: not a toy anymore. The default choice for new greenfield TS backends in many shops.
- ==`tsc` is being rewritten in [[Go]]== ("Project Corsa", Microsoft, announced 2025). Expected ==~10× faster type checking==. Will land progressively.
- ==Zod 4== shipped (2025) with major perf improvements; alternatives (Valibot, ArkType) compete on bundle/speed.
- ==Drizzle has overtaken Prisma== for new projects in many circles.
- ==RSC + Server Actions ([[Next.js]] App Router)== are the dominant new-React pattern.
- ==`import type`==-everywhere is becoming the norm with `verbatimModuleSyntax`.
- The ==edge runtime== (V8 isolates) is steadily eating the long tail of "small backend service."
- Type-level programming is getting ==pushed back on== — overly clever types hurt build time and IDE perf. The community has matured toward "simple types, runtime validation at boundaries."


# What to Learn

1. ==Discriminated unions== + exhaustive `switch` with `never` — the core domain-modeling idiom.
2. ==Generics + constraints== (`extends`) — required for any non-trivial helper.
3. ==Utility types==: `Partial`, `Pick`, `Omit`, `Record`, `Required`, `ReturnType`, `Awaited`, `NonNullable`.
4. ==`unknown` vs `any`== — when each is appropriate.
5. ==[[Zod]]== (or a peer) — parse-don't-validate at boundaries.
6. ==Async patterns==: `Promise.all`, `AbortController`, structured cancellation.
7. ==[[ESM]] vs [[CommonJS]]== — interop is the most painful part of the ecosystem.
8. ==Tree-shaking== and bundle analysis — relevant for frontend & edge.
9. ==`tsconfig.json` flags== — especially `strict`, `noUncheckedIndexedAccess`, `verbatimModuleSyntax`, `isolatedModules`, `module`/`moduleResolution`.
10. ==Monorepo workflow== with pnpm workspaces + Turborepo.


# TypeScript vs Python/Go — Quick Reference

| Concept | Python | Go | TypeScript |
|---|---|---|---|
| Type checking | Optional (`mypy`) | Static, compile-time | Static, compile-time, ==erased at runtime== |
| Errors | `raise`/`try-except` | `return nil, err` | `throw`/`catch` (or Result types) |
| Concurrency | `asyncio` + threads | Goroutines + channels | `async`/`await` + single-threaded event loop |
| Classes | Yes, with inheritance | No (structs + interfaces) | Yes, but structural typing |
| Null | `None` | `nil` (pointers etc.) | `null`, `undefined` (==strict mode forces handling==) |
| Packages | `pip` / `uv` | `go get` + `go.mod` | `pnpm` / `bun` / `npm` + `package.json` |
| Formatting | `black` / `ruff` | `gofmt` (built in) | `prettier` / `biome` |
| Runtime | CPython | Compiled native binary | Node / Bun / Deno / V8 / browser |
| Deploy | Needs interpreter | Single binary | Bundle + runtime, or `bun --compile` |
| Generics | Yes (gradual) | Yes (1.18+) | Yes, ==very expressive== |
| Standard library | Huge ("batteries included") | Large, focused | ==Small== — leans on Web APIs + npm |
