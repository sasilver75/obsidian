
A pattern where ==one service makes an HTTP POST to a URL you control, when an event happens==.
- Instead of you polling them: "Any new orders?", they push to you: "Here's a new order."

> *"An API call is you knowing on their door; a Webhook is them knocking on yours. It's knocking on (e.g.) Stripe's door, handing them a walkie-talkie, and saying "Call me on this (endpoint) when you have something to tell me."*

## How it works:
- You register a URL (a "==webhook endpoint==") with a provider, often by their dashboard or API: 
	- *"POST to https://myapp.com/hooks/stripe when a payment succeeds"*
- When the event fires, the provider sends an HTTP POST with a JSON (or form-encoded) body describing what happened.
- Your endpoint responds with a 2xx; the provider treats non-2xx as a failure and usually retries with backoff.

## Use cases
- Payments (Stripe -> "charge succeeded")
- Source control (Github -> "PR opened", "push to main")
- Communication (Slack -> "user sent message", Twilio -> "SMS received")
- E-Commerce (Shopify -> "order created")
- CI/CD triggers, calendar updates, form submissions (Typeform, Tally)

## Common patterns and gotchas:
- Signature verification
	- Providers sign the payload with a shared secret ([[Hash-based Message Authentication Code|HMAC]]) so you can verify it's really them. ==Always verify==, otherwise anyone who knows your URL can forge events!
- [[Idempotency]]
	- Webhooks are typically at-least-once. The same event can arrive twice (network retry, provider bug). Use the event ID to deduplicate.
- Respond fast, work async
	- Most providers time out at 5-30s. A standard pattern is to validate the signature, enqueue the work ([[Amazon SQS|SQS]], similar), and then return a `200` and process the work associated with the webhook call in the background.
- Ordering is not guaranteed
	- Events may arrive out of order. Use timestamps or version numbers in the payload, not arrival order.
- Retries
	- Non-2xx triggers retries with exponential backoff from the provider for hours or days. Make sure that the failures are real failures.
- Reply/backfill
	- Good providers (Stripe, Github, Shopify) let you replay events from a dashboard when your endpoint was down.

## Alternatives
- [[Polling]]: Simple, but wasteful and laggy. Webhooks replace polling.
- [[WebSockets|WebSocket]]: Bidirecitonal, persistent, in-process. Webhooks are stateless one-shot HTTP from server to server.
- [[Server-Sent Event]]: Server -> browser push. Webhooks are server -> server.
- Message queues ([[Amazon SQS|SQS]], [[Kafka]]): Usually used for internal eventing. Webhooks are the external/cross-org version.

## Tooling
- Svix, Hookdeck, nngest: Managed webhook delivery (if you're the one sending) and receiving (queueing, retries, replay).









