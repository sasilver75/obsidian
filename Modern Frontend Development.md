Checking in on modern rontend development in 2024.


# Runtime Layer
- [[Node.js|Node]] is still the default runtime, v24.15.0 is latest LTS, v25 is out, though.
	- [[ECMAScript Modules|ESModules]] (import/export) has fully displaced [[CommonJS|CJS]] (require0)
- [[Bun]] is the fast Node-compatible runtime, package manger, bundler, test runner all-in one, and is getting real production usage.
	- Deno is an alternate, security-first runtime, less adopted but still alive.
- Typescript is now the default, not optional.

# Package Managers
- [[npm]]: Default, fine, slow
- [[pnpm]]: Content-addressable store, hard-links into `node_modules`, most popular choice for new projects.
- `bun`: Bun's package manager is by far the fastest, and safe to use even if you run on Node.

# Build Tools, Bundlers
- Webpack: Still around, but legacy for new projects.
- [[Vite]]: The default for [[Single Page Application]] (SPA)/library development.
	- Uses [[esbuild]] for dev, and [[Rollup]] for production builds.
		- [[Rolldown]] is the Rust-based rewrite of Rollup, currently (2026) used by Vite.
	- Nearly instant [[Hot Module Replacement]] (HMR)
- `esbuild`: A Go-based bundler/transpiler, very fast.
- [[SWC]]: Rust-based transpiler replacement for Babel, used by [[Next.js]].
- [[Turbopack]]: Vercel's Rust bundler, replacing Webpack inside Next.js.
Trend: ==Everything is being rewritten in Rust for speed.==

# Frameworks
- A "framework" today is typically a meta-framework, a framework wrapping React/Vue/Svelte 
- View Libraries (the rendinrendering core):
	- React: Still the giant; owned by Meta, but [[Vercel]] and the Next.js team drive much of the direction now.
	- Vue: Large, mature, popular outside the US
	- Svelete: Compile-time framework, no virtual DOM, excellent DS.
	- ...
- Meta-frameworks (what you actually build apps in)
	- [[Next.js]]: Dominant React meat-framework; App Router is the modern API.
	- Remix: React meta-framework  focused on web fundamentals (forms, loadres)... merged with React Router v7, which itself a framework.
	- TanStack Start: Newer React meta-framework from Tanstack folks
	- SveleteKit: Svelte's meta-framework
	- Nuxt: Vue's
	- Astro: Content-first, ships zero JS by dfeault, "isladns architecture" (sprinkling interactive components into static HTML)
	- [[Expo]]: React Native meta-framework. If youw nat mobile, this is the answer, not bare [[React Native]]

What's changed in React since 2019?
In 2019 you had Hooks (new), class components (dying), [[Redux]]. Since then:
- Concurrent rendering: React can interrupt, pause, and resume renders, which powers a lot below.
- [[Suspense]]: Declarative loading states... \<Suspense fallback={...}\> waits for child data.
- useTransition/useDeferredValue: Mark updates as low-pri so the UI stays responsive.
- Server Components ([[React Server Components|RSC]]): Components that run ONLY on the server, render to a serialized format, never ship JS to the client. ==This is the biggest paradigm shift since Hooks==, default in Next.js App Router.
	- Mental model: Server Components are rendered HTML + a tree blob; client components are old-school React.
- "use client", "user server directives: File-level boundaries between server and client code...
- ==Server Actions==: Server functions that you can call from client components like RPCs, which handle the network for you. Replaces a lot of `fetch('/api/...)` boilerplate.
- `use()` hook: Unwraps a promise inside a component, integrates with Suspense.
- `useOptimisitc`: Built-in optimistic UI for mutations.
- useFormStatus, useActionState
- React Compiler: Auto-memoizes components so you stop hand-writing useMemo, useCallback...

If you learn one thing: ==server components + server actions are the new mental model==, which ==makes full-stack feel less like "frontend talks to the API" and more like "one program, some pieces run on the server."==


# Rendering Strategies (Vocabulary)
- [[Client-Side Rendering]] (CSR): ==The old SPA model==, send empty HTML+JS, render in browser. ==Bad for SEO and TTI==.
- [[Server-Side Rendering]] (SSR): Render HTML on the server, per-request
- [[Static Site Generation]] (SSG): SSG, but pages can revalidate on a schedule, or on demand.
- Streaming SSR: Flush HTML to the browser as it's ready, instead of waiting for the whole page.
- PPR (Partial Prerendering): In [[Next.js]]: static shell + streamed dynamic holes in one response.
- [[Hydration]]: Client JS attaches event handlers to server-rendered HTML.
- Islands (Astro): Static HTML with isolated interactive components.

# Routing
- File-based routing: Folder structure = URL structure. ==This is the universal pattern now==, e.g. is Next.js App Router

# Data Fetching and state
- The ==old== answer was [[Redux]] + Thunks + Axios + useEffect. ==That's all wrong now.==
- Server Fetching: In App router, you `await fetch(....)` directly in a server component. No client state needed for server data.
- [[TanStack Query]]: FKA React Query, the dominant lcient-side data layer, handles caching, refetching, mutation, optimisitc updates. Use this for anything client-fetched.
- [[SWR]]: Vercel's ligheter alternative.
- [[tRPC]]: End-to-end typesafe RPC. You call server functions like local functions, types flow through. Very popular in the T3 stack.
- GraphQL clients like Apollo, urql, Relay have less hype than in 2019; REST + tRPC took the share.


# Client State (genuinely-client UI state, not server data)
- [[Zustand]]: Tiny, hooks-based, the new default for "I just want a store."
- Jotia, Valtio, Redux Toolkit, React Context, Signals... Just use Zustand.


# Forms
- [[React Hook Forms]]: Uncontrolled, fast dominant.

# Styling:
- The biggest shakeup since 2019.
- [[Tailwind CSS]]: Utility-first CSS, Dominant. v4 (2025) is config-less, CSS-driven, much faster,
- [[CSS Modules]]: Scoped CSS, still fine.
- CSS-in-JS: `styled-components`, `Emotion`: Falling out of favor due to RSC incompatibility and runtime cost.


# Type-safe Stack Glue:
  - [[Zod]] — runtime schemas → TS types. Ubiquitous.
  - [[tRPC]] — typesafe RPC over HTTP.
  - [[Drizzle]] ORM — typesafe SQL builder, lightweight, popular alternative to Prisma. SQL-shaped.
  - [[Prisma]] — schema-first ORM, generates a client. Heavier, more magic.
  - Kysely — typesafe SQL query builder, no ORM.


# Auth
- [[Auth.js]]: Formerly NextAuth, open-source, framework-agnostic.
- Supbase Auth: Bundled with [[Supabase]]
- Clerk: Paid, SaaS, drop-in, native vercel integration.

# Testing:
- [[Vitest]]: Vite-native test runner, replacing [[Jest]] in new projects, with Jest-compatible API.
- [[Jest]]: Still everywhere, slower.
- [[Playwright]]: Microsoft's E2E browser test runner, default choice now.
- [[Storybook]]: Used for component sandbox/visual testing.
Cypress is older E2E, declining.


# Linting, Formatting
- [[ESLint]]: Linter, still standard.
- [[Prettier]]: Formatter, still standard
- [[Biome]]: Rust-based all-in-one formatter/linter, faster.


# Monorepos
- Turborepo: Vercel's build orchestrator + cache

Hosting and deployment:
- [[Vercel]]: Next.js native (also runs anything). 
- [[Netlify]]: Competitor, similar shape
- Cloudflare Pages/Workers: Edge-first, very cheap. The JS runtime is not full Node.


# Mobile/Cross-Platform
- [[React Native]]: Still alive
- [[Expo]]: Meta-framework over REact Native. Use this. Expo Router mirrors Next.js App Router.
- React Native Web: Lets you share a codebase between web and mobile.
- [[Electron]]: Still the industry standard for desktop apps
- [[Tauri]]: The Rust-backed Electron alternative for desktop.

# Trends to Internalize:
1. Server is back. 
	1. [[React Server Components|RSC]]
	2. Server Actions
	3. Streaming
	4. The "thick client SPA + JSON API" pattern is no longer the default for web applications.
2. 