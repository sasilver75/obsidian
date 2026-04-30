A TypeScript runner, "TypeScript Execute". 
- Executes `.ts` files directly without a separate compile step
- Is a  thing wrapper around ==esbuild== and ==[[Node.js]]==.
	- esbuild: 

Think of it as `node`, but it understands TS and modern ESM/CJS interoperation out of the box.

What it does:
- Wraps nodes and registers a `loader` that intercepts imports of `.ts`, `.tsx`, `.mts`, `.cts` files
- Compiles then on the fly with `esbuild` (very fast, ~10-100x faster than `tsc`). Caches the output.
- Strips types, does not type check (if you code has type errors `tsx` will happily run it.)
- Handles [[ECMAScript Modules|ESModules]]/[[CommonJS|CJS]] interop, [[JSX]], top-level await, .json imports without config.
- Forwards Node flags through


How people use it:
```
- Run a script: tsx scripts/foo.ts
- Watch mode: tsx watch src/index.ts (re-runs on file change — common dev-server replacement for backend
code)
- REPL: tsx with no args drops you into a TS-aware REPL
- As a node shim in package.json:
	{ "scripts": { "dev": "tsx watch src/server.ts" } }

```


VS Alternatives:
- vs `ts-node`: tsx is faster and has fewer ESM footguns; most projects have moved to tsx
- vs `tsc` and `node dist/foo.js`: `tsc` builds an output directory, useful for production. `tsx` skips the build, ideal for scripts and dev.
- vs `bun run`: Bun runs TS natively too and is even faster, but it's a different runtime. `tsx` keeps you on Node.

> `tsx` is ==for *running* TS during development and for one off scripts==. It is not a type checker, and is ==not a production builder==.
- Instead, you'd use something that produces optimized output ahead of time. If you're building a frontend/full-stack app, the framework typically picks the build for you. If you're using [[Next.js]], `next build` uses Tubopack under the hood. If you're using [[Vite]], `vite build` runs Rollup for the prod build, `esbuild` for transforms.





