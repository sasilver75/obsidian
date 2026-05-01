---
aliases:
  - WASM
  - WASI
  - WebAssembly System interface
---
A portable binary instruction format, basically a compact, fast bytecode, designed as a compilation target for languages *other than JavaScript!*

The idea: ==Write code in Rust, C, C++, Go, Zig, etc.==, compile it to a `.wasm` binary, and run it in a sandbox at near-native speed.
- Originally developed ==for browsers== (so you could ==ship non-JS code to the web==), but it's since spread to servers and edge runtimes.

>*"Think of WASM as =="a portable, sandboxed CPU."== You compile a program targeting that CPU. The host (browser, edge runtime, plugin system) supplies the operating environment by handing the program a curated set of imported functions. The program can't do anything the host didn't explicitly allow."*

What makes it interesting:
- Language agnostic; any language with a WASM compile target can run wherever WASM runs.
- Fast: Closer to native code than interpreted JS.
- Sandboxed by default: No filesystem, network, or syscalls unless explicitly granted by host. Security model is "deny everything, host explicitly imports capabilities"
- Tiny and portable: Runs identically on any host with a WASM runtime
- Fast [[Cold Start]]s: Milliseconds, versus hundreds of ms for [[Container]]s
	- ==This is why edge platforms love it!==


# Exmaples
- In the browser: Heavy compute that JS is bad at:
	- Figma's rendering engine
	- Photoshop on the web
	- Google Earth
	- Game engines (Unity, Unreal)
- On Server/edge
	- Fastly Compute@Edge
	- Cloudflare Workers (partially)
	- Vercel (uses WASM in some places)
- Elsewhere:
	- Plugin systems, like [[Envoy]] proxy filters, [[Istio]] extensions, Figma plugins, Shopify Functions. Safer than loading native `.so` files, faster than embedding a scripting language.

