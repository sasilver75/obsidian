---
aliases:
  - Ports and Adapters
  - Clean Architecture
---
References:
- Netflix Tech Blog: [Ready for Changes with Hexagonal Architecture](https://netflixtechblog.com/ready-for-changes-with-hexagonal-architecture-b315ec967749) (2020)

A software design style that ==keeps the core business logic isolated from external systems==.

> Business logic should not know known about whether it is being driven by an HTTP request, a test, a CLI command, or a scheduled job.

>"An architectural style where the core application/domain logic is isolated from external systems like databases, APIs, message queues, file systems. The core defines "ports" that are interfaces describing what it needs or exposes. External technologies implement those ports as "adapters." This keeps business rules independent from infrastructure, making the system easier to test/change/maintain.

> Hexagonal Architecture is about protecting the business logic from infrastructure details.

The idea:
- The ==inside== contains your domain models and application rules
- The ==outside== contains databases, web frameworks, message queues, APIs, CLIs, UI, etc.
- The inside talks to the outside through ==ports==, which are interfaces/contracts
- The outside implements those ports through ==adapters==.

So instead of your business logic depending directly on Postgres, Express, React, Stripe, or a REST client, it depends on abstractions like:

```
interface PaymentProcessor {
	charge(amount: Money): Promise<PaymentResult>
}
```

Then Stripe, a mock implementation, or any other provider can be plugged in as an adapter.

The main goal is to make the core of the application easier to test, change, and reason about, because external details are kept at the boundary.

![[Pasted image 20260609195202.png]]


-----------

# Netflix Tech Blog: [Ready for Changes with Hexagonal Architecture](https://netflixtechblog.com/ready-for-changes-with-hexagonal-architecture-b315ec967749) (2020)


- Use Hexagonal Architecture when the edges are likely to change: databases, APIs, protocols, external services, monolith-to-microservice migrations
- Keep business logic in interactors/use cases, not controllers, jobs, API clients, or ORM models
- Treat repositories as interfaces owned by the application core
- Treat databases, REST clients, GraphQL clients, CSV files, etc as data source adapters that implement those interfaces.
- Make transport replaceable too: HTTP controllers, events, cron jobs, and CLI commands should all be able to trigger the same use case.
- Test mostly at the interactor/use-case layer with mocked repositories, add a smaller number of adapter tests (e.g. HTTP integration tests), and a thin set of end-to-end wiring tests.

Good pattern: ==Hide awkward downstream limitations inside the adapter==. If an API lacks a bulk-fetch endpoint, your repository can still expose `getMany(ids)`, while the first adapter implementation makes multiple calls. Later, when bulk fetch actually exists, only the adapter changes.


