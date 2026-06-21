---
aliases:
  - AWS EventBridge
  - EventBridge
---
AWS's managed [[Serverless]] (read: managed) event-routing service.
It lets AWS services, your own apps, and some external SaaS providers communicate by emitting events and having EventBridge route those events to interested consumers.

>  EventBridge is a content-based event router for event-driven systems.

Without EventBridge, a service that detects something happened often has to directly call every downstream system that cares.
- If an `OrderService` creates an order, it might need to verify a `BillingService`, `FulfillmentService`, `EmailService`, etc. If `OrderService`  calls them all directly, OrderService becomes tightly coupled to every consumer.

A common pattern is:
```
Producer -> EventBridge -> SQS queue -> Worker service
```
- So it fills a similar role to [[Amazon SNS|SNS]].
	- SNS: "Publish this message to everyone subscribed to this topic."
		- Use when you want straightforward ***topic-based [[Fan-Out]],*** especially for notifications, webhooks, mobile push, SMS/email, or pushing the same message to SQS queues/Lambda/HTTP subscribers with relatively simple filtering.
		- ==Excellent for notification fan-out.==
	- EventBridge: "Publish this event to a router, and let routing rules decide who should receive it."
		- Use when you are modeling domain or AWS events, and want ***richer content-based routing***, want many event types on the same bus, want AWS events as first-class inputs, want archive/replay, or want the producer to be less aware of the consumer topology.
		- ==Excellent for event-driven architecture and event routing.==

Terms
- An EventBridge event says "something happened."
- An EventBridge rule says "When an event matching this pattern arrives, send it to these targets."
- An EventBridge event bus is the router that receives events and evaluates rules.
- A target is the thing that EventBridge invokes or sends the events to, such as an [[Amazon Lambda|AWS Lambda]] function, [[Amazon SQS|SQS]] queue, [[Amazon SNS|SNS]] topic, another event bus, or an HTTPS API destination.

Flow:
1. Source emits an event
2. The event is sent to an EventBridge event bus
3. EventBridge compares the event against rules attached to that event bus.
4. If a rule matches, EventBridge sends the event to the rule's target or targets.
5. If no rule matches, EventBridge takes no action.
6. If delivery to a target fails for a retriable reason, EventBridge retries. The default is up to 24 hours and up to 185 attempts, with [[Backoff|Exponential Backoff]] and [[Jitter]].
7. You can configure a [[Dead Letter Queue]], usually an [[Amazon SQS|SQS]] standard queue, to retain events that could not be delivered.

Event example
```json
{
  "source": ["com.example.orders"],
  "detail-type": ["OrderCreated"],
  "detail": {
    "paymentStatus": ["AUTHORIZED"]
  }
}
```

