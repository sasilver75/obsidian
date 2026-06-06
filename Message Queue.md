





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
- The worker crashes halfway through






