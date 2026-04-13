---
aliases:
  - React Query
---
Formerly known as React Query.

A tool for handling *server state* -- fetching data from your API, caching it, and refetching when needed.
Without it, every component that needs data from the API has to manually use `useEffect` + `useState` + loading/error states + cache invalidation. It gets repetitive and buggy fast.


Aside on server state vs browser state:
- You might use a tool like [[Zustand]] to handle UI state that lives in the browser (what's the selected hex, what date range is the slider set to, what layers are toggled on)
- You'd use something like TanStack Query to track server state, which would be the actual hex data from your API, dataset metadata, hex detail information, etc. If it came from the server TanStack Query owns it.

