





______________

https://youtu.be/1ISRd0bS714?si=9G-wWShCmGDFuFbh


![[Pasted image 20260606140946.png]]
Imagine that you have a photo sharing app like IG where someone uploads as photo, and then we need to do a bunch of stuff with that photo (resize, filter, content moderation checks).
- Each of these operations takes some number of seconds each.

In the simplest architecture, we might do all of this work synchronously before returning a response to the user after all of this work is done.

This has some real limitations:
- ==Latency==: User stares at a spinner for 10-15s to get confirmation of their photo being uploaded.
- ==Fragility==: What happens if we have a failure doing our filtering? Then the whole upload fails! The user get a message saying that they should retry, and we have to start from scratch.
- ==How do we manage bursty traffic?==: If we get popular on the app store, now we have 5000 a second, rather than 5 a second! Our servers can't handle more than, say 200 a second. So our system is basically falling over under load because it can't handle the throughput.

We can solve these problems with a Message Queue.
![[Pasted image 20260606142001.png]]
- Instead of processing synchronously, when an upload comes in, the server saves the file and writes a message to a queue, which says: `photo 456 needs processing`. Our server can then immediately respond back to the client saying that it's done.
	- ((Does the user do a [[Presigned URL]] upload to an S3 bucket?))
- At the other end of a  queue, you have a pool of worker servers, where each worker pulls a message off of the queue and processing it.
	- ((I think implied here is that each worker is doing all three things (resize, filter, content moderation)))
- Did we fix our problems?
	- Latency is faster for user
	- Fragility: If we have failure, the failure is isolated from the user; if a failure happens, the message is redelivered to another worker
	- Traffic spikes: The queue gets a little deeper; more items in the queue. There's a delay in processing time, but nothing gets dropped.


What is a [[Message Queue]]?
![[Pasted image 20260606142417.png]]
It's just a buffer that sits between the producer, which needs work done, and a consumer, which does the work.
- Producer sends message to queue and totally forgets about it.
- On the other end, Consumers pull the messages off the queue and process them at their own pace.
- The queue is just a buffer between the services that ==decouples== the producers and computers, letting us scale them independently and swap them out transparently.

Think of it like a kitchen: Waiter takes an order, and puts it on the rail for the cooker to make the food. They don't stand there waiting for it to be ready, they go service other tables. So the ticket rail decouples the front of house from the back of house.



How does Message Queues work under the hood?


Acknowledgements
![[Pasted image 20260606142602.png]]
- Worker pulls message off the queue, and starts processing the photo.
- The worker crashes halfway through; what happens to the message?
	- If the queue deleted the message the moment that the consumer grabbed it, then that message would be lost forever!
- When the consumer pulls a message from the queue, the queue doesn't delete it; the consumer has to explicitly send an `ack` back to the message queue saying "Hey, I'm done with this one, you can go ahead and delete it now."
- If the consumer crashes before sending ack, the Queue will eventually assume it wasn't processed, and will redeliver that message to another consumer.

While Worker A is processing a message... and it hasn't acked yet... that message is still in the queue.
What's stopping Worker B from grabbing it too?
- ==Different queueing systems handle this differently==.
	- In [[Amazon SQS|SQS]], when a consumer picks up a message, it becomes *invsible* to other consumers for a configurable window (e,.g. 30 second).
	- In [[Kafka]], they assign each partition to exactly one consumer in a group, so there's no competition in the first place! There's only one consumer eating from each logical queue.
	- In [[RabbitMQ]], they use channel-level prefetch limits and ack timeouts to manage this.

The concept is always the same: Every queueing system needs a way to make sure that a message is only being processed by one consumer at a time.

But there's still a tricky edge case:
![[Pasted image 20260606153120.png]]
- Worker processes the message successfully, but then crashes before it's able to ack.
- The Message queue thinks that the message was never processed, so it redelivers it to another consumer, so that same photo gets processed twice.
	- This might not matter for processing photos, but what if the action what `charge Sam $50`? 
	- Duplicate processing of this message might cost $100!

This problem brings us to [[Delivery Guarantee]]s of message queues, of which there are three common ones:
- [[At Least Once|At-Least-Once-Delivery]]: Message may be delivered more than once.
	- The most common delivery guarantee. Each message will be delivered at least one time, but it might be delivered more than one time.
	- Your consumers now need to be [[Idempotency|Idempotent]], so that processing the same message twice gives the same result as processing it once.
		- Use natural idempotency: `Set balance=5` is idempotent. `Set balance -= 5` is not.
		- Alternatively, see [[Idempotency|Idempotency Key]]s to check if actions have already been completed.
		- ==Almost always the right answer during a system design interview.==
- [[At-Most-Once Delivery]]: Fire and forget; message might never arrive/be processed.
	- Consumer takes a message off the queue, and we immediately delete it off the queue at that moment. If the consumer crashes, so what!
	- Only want to use this for analytics/metrics where losing some datapoints is acceptable.
- [[Exactly Once|Exactly-Once-Delivery]]: Every message is processed exactly one time.
	- True exactly-once is ==extremely hard to deliver.==
	- [[Kafka]] supports *a form of it,* but it comes with real tradeoffs and limitations. Don't promise exactly-once unelss you can clearly explain the mechanism and defend it.


So when should you reach for a queue? There are four signals that might make you look:
1. ==Async Work==: When a user doesn't need an immediate result (sending emails, generating reports, processing uploads)
2. ==Bursty Traffic==: Absorbing spikes in traffic without dropping requests by using a queue to smooth out the load, accumulating a backlog of work that consumers can get to when they can.
3. ==Decoupling==: Letting services scale and fail independently. This is when your producer and consumer might have totally different scale/hardware needs. Upload services might be lightweight, while workers might need GPUs or beefy machines with lots of memory. With a Queue, you can scale/provision each side independently.
4. ==Reliability==: When you can't afford to lose work. If a downstream service is temporarily unavailable, the queue holds onto that message until it comes back online.

==Be careful introducing a queue into what should be a synchronous workflow.==
- If you have a strict latency requirement, like <500ms response times... by adding a queue, you're nearly guaranteeing that you're going to break that constraint.
	- There's complexity in figuring out how to get that notification back to the user, and the network hops hurt.


#### Common Deep Dives on Queues

![[Pasted image 20260606154851.png]]
Q: How does your queue handle increasing throughput?
- A single queue can only handle so much. 
- When you need more, you [[Partition]] the queue into independent sequences/sub-queues of messages. Different works can then process different partitions in parallel, so that throughput can scale horizontally with the numbero f partitions.
	- ((I believe that typically you have multiple hosts/machines that each have partitions))
![[Pasted image 20260606155000.png]]
On the consumer side, you have [[Consumer Group]]s, which are pools of workers that divide the partitions amongst themselves. If you have 6 partitions and 3 consumers in a group, then each consumer can handle 2 partitions.
- There IS a ceiling here. You can't have more consumers than you have partitions! ((I think this is pretty Kafka-specific)) If you have 6 partitions and you have 6 consumers, adding a seventh won't help.

If you need more throughput, you can add more partitions.
If you need to go faster, you can add more consumers.

The [[Partition Key]] is an important choice; it's analogous to choosing a [[Shard Key]] in a database:
1. ==Ordering==: Messages with the same partition key always go to the same partition, and within a partition, ordering is guaranteed.
	1. If a user deposits $100 and then withdraws $50... clearly, they should be able to do this. If those two messages ended up on different partitions, then the withdrawal could get processed first, and be rejected! By using `account_id` as the partition key, both messages land on the same partition, are in the right order, and are processed in the right order.
2. ==Even Distribution==: You want to spread your work across these partitions pretty evenly.
	1. IF you have a ride-sharing application and are partitioning by city, then your NYC partition is going to be slammed like Boise sits there getting less activity. This is a [[Hot Spot|Hot Partition]]! You probably want to partition by something more evenly distributive like a `ride_id`.

==TRADEOFF==: The key that gives you the appropriate ordering might not always be the key that gives you the best distribution of work! Talking about this tradeoff is important.



Another Problem: what if your producers outpace your consumers?
![[Pasted image 20260606155701.png]]
This isn't a burst of traffic, the queue is growing and growing and growing.
Eventually you'll run out of memory on that queue, and things will start to go wrong.

What can you do?
1. [[Autoscaling]], where you monitor the ==queue depth== and when it starts to grow too much, you spin up more consumers or add additional partitions to the queue.
2. Apply [[Backpressure]] to the producers themselves, slowing down the producers by rejecting messages or returning an error to the client saying "Hey, we're a little overloaded right now, try again in a minute"
3. Set Monitoring and [[Alert|Alerting]] on your queue depth so that you know when something like this is happening.

Interviewers want to know that you understand that a queue is not magic.


Another problem: Sometimes a message just fails to process! Maybe an image file is corrupted or a downstream service is unavailable, for instance.
![[Pasted image 20260606155956.png]]
This will never succeed, no matter how many times you retry, or continue to fail. These are called [[Poisoned Message]]s, which are malformed or problematic message that crashes the consumer every time, and there's no recovering from it. 
- Without guardrails, it will keep failing retries, and will be doing [[Head-of-Line Blocking]], using the resources of the consumers indefinitely.
- Most queueing systems let you configure a Max Retry count (e.g. 5 times), and if it still hasn't succeeded after that, you shunt it to a [[Dead Letter Queue]] (DLQ), which is a separate queue where failed messages go, so that an admin can inspect them later and figure out what went wrong, meanwhile the main queue keeps moving.

Mentioning this proactively in an interview is great.



Another problem: What happens in the Queue itself goes down?
![[Pasted image 20260606160238.png]]
- Modern message queues like [[Kafka]] persist messages to disc, and they can replicate them across multiple brokers.
	- If one server goes down, no messages are lost.
- In Kafka, they store messages on discs with configurable retention windows. You can keep message around forever if you wanted to, so you can actually replay messages from the past too, which is a very powerful tool (e.g. for a recovery scenario).




Let's go through some of the most common technologies ([[Amazon SQS|SQS]], [[Kafka]], [[RabbitMQ]])
- If you don't have a default already, choose [[Kafka]], it's kind of the interviewing industry standard, which is probably the most widely used


[[Kafka]] (Choose Me!)
- Distributed streaming platform that can act as both a [[Message Queue]] and a [[Stream Processing]] platform.
- It's durable because it writes data to disk.
- Scales via adding partitions
- Supports consumer groups
	- Multiple consumer groups can read from the asme data independently, and you can replay messages if you need to.
- UNIQUE: Messages *are not removed after they're consumed!* Instead, consumers maintain their own offsets in the partition log. Messages stick around for a configurable duration.


[[Amazon SQS]]
- The AWS managed version of a message queue. It's simple, fully managed, no infrastructure you need to worry about.
- Comes in two flavors:
	- Standard queue: Gives best-effort ordering, but with very high throughput.
	- FIFO queues: Gives strict ordering, but at a lower throughput.
- Great choice when you want something straightforward, you don't need advanced features, and the 


[[RabbitMQ]]
- A more traditional message broker, supports complex routing patterns through what it calls exchanges and bindings. 
- It's all the same underlying concepts: It's less common.... It's worth knowing if it exists.















