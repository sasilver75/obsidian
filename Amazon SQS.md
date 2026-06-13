---
aliases:
  - SQS
  - AWS SQS
---
AWS's managed message queue service: A managed, durable queue where producers enqueue messages, and consumers poll/process/delete them.

SQS decouples producers from consumers. A web server, for example, can accept an image upload and put a message like `resize image 123.jpg` onto an SQS queue for a background later to read and do expensive resize work.

Often used mechanically like this:
```
1. Web app receives request.
2. Web app sends message to SQS.
3. Worker polls SQS.
4. Worker receives message.
5. SQS hides message during the visibility timeout.
6. Worker processes message.
7. Worker deletes message.
8. If worker fails before deletion, message reappears for retry.
```

Note that if a worker successfully performs processing of an item but crashes before deleting the SQS message, the message will be retried by another worker that claims it. As a result, your application needs [[Idempotency]] for real world side-effects.

# Standard Queue vs [[First-In First-Out|FIFO]] Queue
SQS has two main queue types:
- ==Standard Queue==: Best-effort ordering, best for high-throughput background work where duplicates and out-of-order processing are acceptable. Prioritizes scale and availability.
- ==FIFO Queue==: Strict ordering within each MessageGroupId ((Partition key, likely)), best for workflows where order matters, such as account events or payment state transitions.  You have to design around message groups and throughput constraints.


# Usage in Combination with [[Amazon SNS]]
- One of the most classic design patterns in AWS architecture is using SNS plus SQS for ==durable fanout.==
	- SNS answers: "Who should be notified that this event happened?"
	- SQS answers: "How can each receiver process that event reliably, at its own pace?"

1. A producer publishes one message to an SNS topic.
2. Each subscribed SQS queue receives its own copy of the message.
3. Each downstream service polls its own SQS queue.
4. Each service processes and deletes messages independently.
5. If one service is slow, offline, or broken, the other services can keep processing their own queues.

The last point is the real reason why this pattern is so common: SNS delivers the event to multiple subscribers, while SQS gives each subscriber a durable backlog. SNS is the broadcast layer, and SQS is the durable per-consumer buffer.

Use **SQS directly** when one queue represents one work stream.
Use **SNS + SQS** when one event should be delivered to multiple independent consumers.
- (Largely because it's not like for SQS you can have a durable log with multiple consumer groups that each can feed off the log without deleting entries from the broker. In SQS, when a message is acked, it's deleted.)










