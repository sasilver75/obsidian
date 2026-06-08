
The general term for client-side storage mechanisms that let a website save data in the user's browser.

Typically, this storage is scoped by ==origin==, meaning scheme, host, and port, such as `http://example.com` (80 is the default port for HTTP, 443 for HTTPS).

Different storage APIS vary by:
- How long data lasts
- Whether JavaScript can access it
- Whether it is sent to the server automatically
- How much data it can hold
- Whether the API is synchronous or asynchronous
- Whether it is meant for application data, HTTP state, or cached network responses

Some of the storage APIs we cover below:
1. [[Local Storage]]: Use for tiny persistent preferences.
2. [[Session Storage]]: Use for temporary, per-tab state.
3. [[Cookie]]s: Use for server-visible state, especially [[Authentication|Authn]] [[Session]]s with `HttpOnly`, `Secure`, and `SameSite`
4. [[IndexedDB]]: Use for structured, powerful client-side data storage of many datatypes
5. [[Cache API]]: Use for offline and network response caching

# [[Local Storage]]
- `localStorage` is a simple key-value store for string data.
- It persists across page reloads, browser restarts, and tabs until the user or app clears it.
- It is origin-scoped and NOT automatically sent to the server.
- It is synchronous, so heavy use can block the main thread.
- GOOD for:
	- Theme preference
	- Simple UI settings
	- Small bits of non-sensitive application state
- AVOID using it for:
	- Auth tokens
	- Large data
	- Frequently updated


# [[Session Storage]]
- `sessionStorage` is similar to `localStorage`, but its lifetime is tied to a single browser tab or window.
- Survives reloads, but is cleared when the tab closes. It is NOT shared across tabs, even for the same site.
- Like local storage, it stores strings and is synchronous, so heavy use can block the main thread.
- GOOD for:
	- Temporary form state
	- Per-tab wizard progress
	- Short-lived UI state

# [[Cookie]]s
- Small pieces of string data that, unlike [[Local Storage]] or [[Session Storage]], are automatically attached to [[HTTP]] requests to a server on every matching request.
- cookies are small, usually around 4KB each, and have important attributes:
	- `HttpOnly`: Prevents JS access
	- `Secure`: Only sent over [[HTTPS]]
	- `SameSite`: Controls cross-site sending behavior
	- `Expires` / `Max-Age`: Controls lifetime
- For [[Authentication]], a secure `HttpOnly` cookie is usually safer than putting a token in [[Local Storage]], because [[Cross-Site Scripting|XSS]] cannot directly read it.
- Commonly used for:
	- Login [[Session]]s and server-side [[Authentication]]
	- Personalization
	- Tracking and analytics, though this is heavily regulated and restricted by modern browsers

# [[IndexedDB]]
- The browser's built-in client-side database.
- It is asynchronous, transactional, origin-scoped, and can store much more complex data than [[Local Storage]], including objects, arrays, blobs, files, and indexed records.
- It's more powerful than previous options but also more verbose to use directly; many apps use a wrapper library around it.
- GOOD for:
	- Offline-capable applications
	- Large client-side datasets
	- Drafts
	- Sync queues
	- Structured application data
	- Cached API data that needs querying


# [[Cache API]]
- The Cache API stores HTTP Request-Response pairs. 
- It is commonly used with [[Service Worker]]s to make apps load faster or work offline.
- It's not primarily a general-purpose dat store -- use [[IndexedDB]] for structure application data, and use the Cache API for network resources.
- GOOD for:
	- Caching static assets
	- Offline pages
	- API response caching
	- [[Progressive Web Application]]s



