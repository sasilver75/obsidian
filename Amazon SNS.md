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

How SNS actually delivers to a subscriber depends on the specific subscription protocol. Each subscription to an SNS topic says: "When a message is published to this topic, deliver it to this endpoint using this protocol."

More mechanically, an SNS subscription has:
1. A topic [[Amazon Resource Name|ARN]], meaning the SNS topic being subscribed to.
2. A protocol 
	1. `http`: endpoint value is a public `http://` URL, sends a `POST`
	2. `https`: endpoint is a public `https://` URL, sends a `POST`
	3. `lambda`: endpoint is a Lambda function ARN, invokes the Lambda function
	4. `sqs`: endpoint is a SQS ARN, sends a message to the queue
	5. `email`: endpoint is an email address, sends a text email
	6. `email-json`: endpoint is an email address, sends a JSON-formatted email
	7. `sms`: endpoint is a phone number, sends a SMS message
	8. `application`: endpoint is a mobile platform endpoint ARN, sends a mobile push notification
	9. `firehose`: endpoint is a [[Amazon Data Firehose|Data Firehose]] ARN, writes to a Firehose stream
3. An endpoint, such as a SQS queue [[Amazon Resource Name|ARN]], a Lambda function [[Amazon Resource Name|ARN]], URL, email address, or phone number.
4. Optional deliver behavior, such as filter policies, retry behavior, raw message delivery, or [[Dead Letter Queue]] sessions.

==Importantly, not any arbitrary thing can be a subscriber!== SNS supports a fixed set of subscription protocols and endpoint types. If you want an unsupported system to receive SNS messages, the usual pattern is to put something supported in between, such as a Lambda function or SQS queue.

Registering SNS subscription for Lambda:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:123456789012:order-events \
  --protocol lambda \
  --notification-endpoint arn:aws:lambda:us-east-1:123456789012:function:process-order-event
```
Registering SNS subscription for SQS:
```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:123456789012:order-events \
  --protocol sqs \
  --notification-endpoint arn:aws:sqs:us-east-1:123456789012:fulfillment-queue
```

Example publishing to the SNS topic:
```bash
aws sns publish \
  --topic-arn arn:aws:sns:us-east-1:123456789012:order-events \
  --message '{"eventType":"OrderCreated","orderId":"ord_123","total":49.99}' \
  --message-attributes '{
    "eventType": {
      "DataType": "String",
      "StringValue": "OrderCreated"
    }
  }'
```
If, for example, your Lambda needs "arguments," you put those in the published SNS message, then the Lambda parses the SNS message and treats those fields as its application-level inputs.

SNS should not need to know your Lambda function's parameter list; SNS only knows how to invoke Lambda with an SNS event envelope. Your Lambda handler adapts that event envelope into your domain-specific function call:
```js
// Your actual registered Lambda function handler
export async function handler(event) {
  for (const record of event.Records) {
    const message = JSON.parse(record.Sns.Message);
    await processOrderCreated(message);
  }
}

async function processOrderCreated({ orderId, customerId, total }) {
  // normal application code here
}
```


# Subscriber Message Filtering
-   Rather than every subscriber receiving every message, SNS supports subscription filter policies — JSON policies that define which messages a subscriber receives, based on message attributes. Different subscribers can have different filters on the same topic.
- This avoids having to create separate topics for every message type (one topic, filtered subscriptions)

SNS makes [[At Least Once]] delivery attempts. For HTTP endpoints it retries with exponential backoff on failure. For SQS and Lambda, delivery is highly reliable since those services are AWS-managed. For email and SMS, delivery depends on external systems.                                                                                                    
[[Dead Letter Queue]] (DLQ) — if delivery repeatedly fails, SNS can route undeliverable messages to an SQS DLQ for investigation and reprocessing.


# SNS vs [[Amazon EventBridge|EventBridge]]
- AWS eventBridge is the enwer, more powerful event bus service, and is ==generally prefererd for new architectures needing sophisticated event routing.== SNS remains common for simple fan-out and mboile push notifications, where its direct delivery model is appropriate.