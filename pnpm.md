A fast, disk-efficient package manager for JS/[[Node.js]] projects, and an alternative to [[Node Package Manager|npm]] and yarn.

Key differences from `npm`:
- Hard links and symlinks: ==Packages are stored once in a global content-addressable store== and hard-linked into `node_modules` so 10 projects using React don't store 10 copies.
- Strict `node_modules`: Only packages you explicitly declare in `package.json` are accessible, no phantom dependencies from hoisting.
- ==Faster installs==: Especially on repeat installs across projects, since it reuses the global cache.
- Workspaces: First-class monorepo support via `pnpm-workspace.yaml`



#### What is `node_modules?`
- A directory that JS package managers (npm, pnpm, yarn) create in your project to store all installed dependencies and their dependencies.
- When you run (e.g.) `npm install`, it reals `package.json`, downloads the required packages, and puts them in `node_modules/`, so your code can import or require them.
	- This directory is always `.gitignore`'d, and can get large (500MB-1GB)
	- Packages' dependencies also live here, which is why there are often hundreds of folders inside.
- During `npm install`, the package manger resolves the full dependency tree (your deps, their deps, their deps deps), downloads tarballs from the npm registry, and writes a lockfile (`pnpm-lock.yml`, e.g.) pinning exact versions so that every install is identical. It places the code in `node_modules/` so that `import React from 'react'` resolves correctly.

