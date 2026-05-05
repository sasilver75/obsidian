---
aliases:
  - SPM
---
Swift Package Manager (SPM) is Apple's official dependency and modularization tool, analogous to Go modules.

A [[Swift]] package is a directory with a `Package.swift` manifest that declares:
- Its source files
- What it depends on (other packages, by URL or local path)
- What it exposes (one or more library targets)

Two flavors are relevant:
1. Remote SPM packages: A third-party dependency, fetched from Git URL.
2. Local SPM packages: Your own code, factored into a sub-directly inside the repo, with its own Package.swift.
	- Q: Why split into local SPM modules at all?
		- ==Compile-time isolation==; Touching `APIClient` only rebuilds `APIClient` and its dependents, not the whole app. This matters for big projects.
		- ==test isolation==; Each package has its own test target; you can run `swift test` on it without launching the iOS Simulator. Faster TDD loop for pure-logic code.
		- ==Enforced boundaries:== If `DesignSystem` doesn't depend on `APIClient`, the compiler enforces it; no accidental import.
		- Downside cost: Every split adds another `Package.swift` to maintain, slows the initial build a bit, and makes navigation slightly more annoying (jumping between targets).



