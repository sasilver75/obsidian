A type of [[Browser Storage]].

IndexedDB is the browser's built-in client-side database.
- It is asynchronous, transactional, origin-scoped, and can store much more complex data than [[Local Storage]], including objects, arrays, blobs, files, and indexed records.
- It's more powerful than previous options but also more verbose to use directly; many apps use a wrapper library around it.

GOOD for:
- Offline-capable applications
- Large client-side datasets
- Drafts
- Sync queues
- Structured application data
- Cached API data that needs querying
