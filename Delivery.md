
In the context of a queueing system, means:

> A [[Message Queue]], broker, or log has made a message available to a consumer for processing, and the system now treats that consumer or consumer group as having had an opportunity to process it.

Importantly: Delivery does not necessarily mean: "The broker *pushed* bytes into the consumer," it means that the message crossed some boundary from "available work" into "work currently being offered to, fetched by, assigned to, or observed a consumer."

In a pull-based queue, the consumer may initiate the interaction:
```
Consumer: Give me messages
Queue: Here is message M
```
Even though the consumer pulled the message, the queueing vocabulary may still say "*message M was delivered to the consumer.*"

In a lease-based queue like [[Amazon SQS|AWS SQS]], delivery usually means:
> The message was returned to a consumer, and temporarily hidden from other consumers.

Later, if the consumer doesn't *acknowledge* or delete the message before the visibility timeout expires, the queue may deliver the same message again.

In a log-based system like [[Kafka]], deliver is looser:
> A consumer fetched or was assigned records from a partition.

