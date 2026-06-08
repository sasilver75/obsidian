A type of [[Browser Storage]].

`localStorage` is a simple key-value store for string data.

- Persists across page reloads, browser restarts, and tabs until the user or application clears it.
- Scoped by origin: scheme, host, and port.
- Not automatically sent to the server.
- Synchronous, so heavy use can block the browser's main thread.

GOOD for:
- Theme preference
- Simple UI settings
- Small bits of non-sensitive application state

AVOID using it for:
- Auth tokens
- Large data
- Frequently updated data
