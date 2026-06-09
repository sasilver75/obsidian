
A client-side cache is data stored on the user's device or in the client application so that repeated applications can be served locally, instead of fetching again from the server.

It's useful for faster loading, offline support, reduced bandwidth, and less server load. The main risks are stale data, storage limits, and needing clear invalidation rules so that the client knows when to refresh.

Examples:
- Browser HTTP cache ([[Cache API]])
- Various [[Browser Storage]] options ([[Local Storage]], [[Session Storage]], [[IndexedDB]])
- Mobile app disk cache
- In-memory state in a web application (Yep, you can use client-side caching for a server too, though here it would also be fine to call it an [[In-Process Cache]])
- [[Service Worker]] cache


