
A single deployable application ([[Monolith]]) with strong internal boundaries.

It's still one deployable application, but internally it's divided into modules with explicit boundaries: for example, `Accounts`, `Billing`, `Notifications`, `Reporting`, `Admin`, etc.

Each module owns its own logic and ideally its own data access. Other modules interact through well-defined interfaces, events, commands, or service APIs inside the same process. You get much of the design discipline of [[Microservice]]s without paying the full distributed systems tax.

