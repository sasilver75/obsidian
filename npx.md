
npx is a package runner that ships alongside [[Node Package Manager|npm]] since 5.2.
- It's job is to ==execute a package's CLI binary, often without permanently installing it==

If you run `npx vercel deploy`:
1. npx checks if `vercel` is already in your `./node_modules/.bin` (project-local)
2. If yes, it runs that.
3. If no, downloads the package to a temporary cache, runs its binary, and leave it cached for next time.

So `npx` is for *invoking* a tool, whereas `npm install -g` is for *installing* one.

Essentially, `npx` is: "I just want to invoke this CLI; figure out where it lives (local install? global? need to download?) and just run it." It's the ==canonical way to run scaffolders like `create-next-app` (which you only need once, then never again==. Typically adds 1-3s versus running an installed binary directly (checking caches, doing registry fetches), which is annoying if it's something that you're going to be using very frequently.



