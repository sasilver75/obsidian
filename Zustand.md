A small, fast, and scalable barebones state management solution for React, using simplified flux principles. Has a hooks-based API, isn't boilerplatey or opinionated. A simpler alternative to Redux -- it's "global variables done right."

Why Zustand over Redux?
- Simple and un-opinionated
- Makes hooks the primary means of consuming state
- Doesn't wrap your app in context providers
- [Can inform components transiently (without causing render)](https://github.com/pmndrs/zustand#transient-updates-for-often-occurring-state-changes)

Why Zustand over Context?
- Less boilerplate
- Renders components only on changes
- Centralized, action-based state management