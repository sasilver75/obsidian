A type of [[Browser Storage]].

`sessionStorage` is similar to [[Local Storage]], but its lifetime is tied to a single browser tab or window.

- Survives reloads within the same tab.
- Cleared when the tab or window closes.
- Not shared across tabs, even for the same site.
- Stores strings and is synchronous, so heavy use can block the browser's main thread.

GOOD for:
- Temporary form state
- Per-tab wizard progress
- Short-lived UI state
