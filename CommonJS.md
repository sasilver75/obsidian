---
aliases:
  - CJS
---
There are two Module systems in JS, which are how files share code with eachother.

[[CommonJS]] (CJS) is the original [[Node.js]] module system (2009), which is synchronous and dynamic.
```js
  // math.js
  function add(a, b) { return a + b; }
  module.exports = { add };

  // app.js
  const { add } = require('./math');
```
Above:
- require() runs synchronously and returns a value
- You can require() conditionally, inside functions, with computed paths
- Files are .js by default in older Node projects, or explicitly .cjs


Alternatively, [[ECMAScript Modules]] (ESM) is the official JS standard since 2015, with Node adding support in 2019. It's static, and async-capable.
```js
// math.mjs
export function add(a, b) { return a + b; }

// app.mjs
import { add } from './math.mjs';
```
Above:
- import is a top-level statement, statically analyzable
- Enables tree-shaking (bundlers can see exactly what's imported)
- Supports top-level await
- Files are .mjs, or .js when package.json has "type": "module"


Mental model
  - CJS = old, synchronous, require/module.exports, Node-only
  - ESM = standard, static, import/export, browser + Node + everywhere
  - New code should be ESM. You'll keep meeting CJS for years because of legacy deps.


