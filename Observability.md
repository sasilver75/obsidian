



Elie Steinbock uses:
- [[Axiom]] CLI (used to use their [[Model Context Protocol|MCP]]): "I'm a happy paying customer"
- [[Sentry]] CLI
- Database (Gives agents limited access to the database; you *should* be afraid, but make sure that it's read-only access... but that's not enough; if you give it access to the entire database, that might not be okay; so you can create a [[View]] (e.g. in PG) and only have it access that view, which has sensitive information stripped out)
	- This also applies to AWS; if you want to have an agent access your infrastructure, you create an IAM profile with very tightly scoped permissions
- [[PostHog]] CLI if you need it
	- "We don't use it for debugging, but we do use it for analytics"
His point is that AI needs access to your logs so that it can understand what's going on.
