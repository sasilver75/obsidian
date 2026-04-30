---
aliases:
  - npm
---
The default package manager for [[Node.js]], which does three things:

1. Registry: A public database of JS packages @ npmjs.com
2. CLI: The `npm` command for installing, publishing, and managing dependencies
3. `package.json`: The manifest that declares a project's dependencies, scripts, metadata

Common commands:
```bash
npm install — install all dependencies from package.json
npm install <pkg> — add a new dependency
npm run <script> — run a script defined in package.json
npm publish — publish a package to the registry
```

See also: [[pnpm]]
