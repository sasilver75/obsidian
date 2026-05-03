---
aliases:
  - RSC
---
A type of [[ReAct (Agent)]] component that runs ==only on the server==, never on the client. The output is serialized and streamed to the browser, where React reconstructs the UI, but the component's code, dependencies, and data fetching never touch the user's machine.

Thsi buys you:
- Zero JS bundle cost for server components (a 200KB markdown library used in an RSC ships nothing to the client)
- Direct backend access (you can `await db.query(...)` inside the component itself)
- No API route, no `fetch`, no client-side data layer.
- Secrets stay on the server
- Async components

You give up:
- Server components cannot:
	- Use `useState`, `useEffect`, or any hook that needs a runtime.
	- Attach event handlers (`onClick, etc`)
	- Use browser APIs  like `window`, `localStorage`


For interactivity, you drop into a client component by putting `"use client"` at the top of the file.
Client components are the react that you already know from 2019. They ship JS, hydrate, and have state.

# The mental model
Think of it as a tree:
- ==Server components== are the ==default==. They render on the server, produce a serialized tree.
- ==Client components== are ==leaves (or subtrees) marked with "use client".== They get [[Hydration|Hydrated]] in the browser.
- Server components can render client components and pass them props (must be serializable — no
functions).
- Client components cannot import server components, but they can receive them as children props.

# Why it's a big deal
- Before RSC, "React app" meant "SPA that fetches JSON from an API." RSC collapses that: the server renders components that already have their data, and only the interactive parts ship to the client. 
- It blurs the front/back boundary in the same direction tRPC and server actions do.

==Short version: RSC is React's answer to "stop shipping the whole app to the browser when most of it is just rendering data." It's the single biggest architectural shift in React since hooks.==