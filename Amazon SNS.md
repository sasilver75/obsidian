---
aliases:
  - SNS
  - Simple Notification Service
  - AWS SNS
---
AWS's fully-managed [[Publish-Subscribe|Pub Sub]] messaging service, enabling decoupled, event-driven communication between distributed systems by allowing publishers to send messages to topics that fan out to multiple subscribers simultaneously.

==Publishers== send messages to ==Topics==, then SNS delivers that message to all ==Subscribers== of that topic in a ***simultaneous*** [[Fan-Out]]. 
- The publisher *does not know and does not care who the subscribers are, and subscribers don't know who published*; they are decoupled.
- This contrasts with point-to-point messaging (like [[Amazon SQS|SQS]] queues) where one sender sends to one receiver.

## Topics: a named channel through which messages flow. Publishers send to a topic, and subscribers receive from it.
- ==Standard topics==: [[At Least Once]] delivery, best-effort ordering, messages may arrive out of order or more than once. High-throughput.
- ==FIFO topics==: [[Exactly Once]] delivery, strict ordering in a message group. Lower throughput, used when order and deduplication matter (financial transactions, inventory updates).

## Subscriber types
- SNS can deliver to many endpoint types:
	- [[Amazon SQS|SQS]] queue: ==Most common==; SNS fans out to multiple SQS queues, and each queue processes independently. Durable; if consumer is down, messages wait in the queue.
		- This is ==THE canonical pattern in AWS architecture!==
		- SNS alone ==doesn't buffer messages;== if a subscriber is unavailable when a message arrives, that message is lost; ==combining SNS with SQS gives you both fan-out and durability==.
	- [[Amazon Lambda|Lambda]] function: Invoke a Lambda directly on each message; serverless processing.
	- HTTP/HTTPS endpoint: POST the message to a webhook URL
	- Email/Email-JSON: Send email notifications.
	- SMS: Text message to a phone number
	- Mobile push: Push notifications to iOS and Android devices
	- [[Amazon Kinesis|Kinesis]] Data Firehouse: Stream messages to S3, Redshift, OpenSearch

```
Event happens                                                                    
    → Publish to SNS topic
      → SQS Queue A (email service reads and sends notifications)
      → SQS Queue B (analytics pipeline reads and records event)                 
      → SQS Queue C (audit log reads and stores)                                 
      → Lambda (real-time dashboard update)
```

# Subscriber Message Filtering
-   Rather than every subscriber receiving every message, SNS supports subscription filter policies — JSON policies that define which messages a subscriber receives, based on message attributes. Different subscribers can have different filters on the same topic.
- This avoids having to create separate topics for every message type (one topic, filtered subscriptions)

SNS makes [[At Least Once]] delivery attempts. For HTTP endpoints it retries with exponential backoff on failure. For SQS and Lambda, delivery is highly reliable since those services are AWS-managed. For email and SMS, delivery depends on external systems.                                                                                                    
[[Dead Letter Queue]] (DLQ) — if delivery repeatedly fails, SNS can route undeliverable messages to an SQS DLQ for investigation and reprocessing.


# SNS vs [[Amazon EventBridge|EventBridge]]
- AWS eventBridge is the enwer, more powerful event bus service, and is ==generally prefererd for new architectures needing sophisticated event routing.== SNS remains common for simple fan-out and mboile push notifications, where its direct delivery model is appropriate.