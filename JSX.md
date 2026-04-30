A ==syntax extension== to JavaScript that ==lets you write HTML-like markup directly in your code==.
- Created by Facebook for React, but other frameworks (Solid, Preact) use it too.
- `.tsx` is just JSX in a TypeScript file.

```js
function Greeting({ name }) {
    return <h1 className="title">Hello, {name}!</h1>;
}
```
Above:
- That \<h1\>...\</h1\> is not a string and not real HTML — it's JSX.
- JSX is ==syntactic sugar for function calls==. A compiler (Babel, esbuild, swc, tsc) transforms it into plain JS before it runs.
The above becomes:
```js
 function Greeting({ name }) {
    return jsx('h1', { className: 'title', children: ['Hello, ', name, '!'] });
}
```
- The browser never sees JSX, it only sees the compiled JS.




