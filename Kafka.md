SDIAH Video: https://www.hellointerview.com/learn/system-design/deep-dives/kafka

This is a deep dive on Kafka, specifically through the lens of a SD interview
- It's an ==event streaming platform== that can be used as a ==message queue== or as a ==stream processing system==!
- It's used by 80% of the Fortune 100, and it's one of the Top 5 Technologies that hellointerview thinks you should familiarize yourself with to ace system design interviews.
- You need to be able to understand when to use Kafka and have enough depth to be able to speak in detail about some of the tradeoffs necessary to impress your interviewer...

Four Sections:
1. Motivating Example (Moved to bottom so it doesn't clutter things)
2. Overview
3. When to use it in an interview
4. What to know for deep dives


# Overview
- Terminology Introduced:
	- **==Broker==**: A server (physical or virtual) that hold the "queue." Each broker is responsible for storing data and serving clients; The more brokers, the more data we can store and the more clients we can serve.
	- **==Partition==**: The "queue". An ordered, immutable sequence of messages that we append to, like a log file. Each broker can host multiple partitions. A *physical* grouping of messages. Each broker has a number of partitions, which are ordered, immutable sequence of messages that are continually appended to: Think of a log file! Partitions are the way Kafka scales, allowing for messages to be consumed in parallel. ==Each partition functions essentially as an append-only log file!== This log is **immutable**, so that messages cannot be altered or deleted, which simplifies replication, speeds up recovery, and avoids consistency issues. The simplicity of the append-only log facilitates horizontal scaling; more partitions can be added and distributed across a cluster of brokers to handle incerasing loads, and each partitions can be replicated across multiple brokers to enhance fault tolerance.
		- Each partition has a designated **leader replica**, which resides on a broker. This leader replica is responsible for **handling all READ and WRITE requests for the partition**! The assignment of the leader replica is managed centrally by the cluster controller, which ensures that each partition's leader replica is effectively distributed across the cluster to balance the load.
		- Alongside the leader replica, several **follower replicas** exist for each partition, residing on different brokers. These **followers don't handle direct client requests**, instead passively replicating data from the leader replica. They're ready to take over, should the leader replica fail. Followers continuously synchronize with the leader replica to make sure they have the latest set of messages appended to heir partition log.
		- The **Controller** in the Kafka cluster manages this replication process, as well as monitors the health of all brokers and manages the leadership and replication dynamics. **When a broker fails, the controller reassigns the leadership role to one of the in-sync follower replicas** to ensure continued availability of the partition.
	- ==**Topic**==: A logical grouping of partitions. You can publish to and consume from Topics in Kafka. A *logical* grouping of messages (across physical/virtual servers, each hosting a partition for a given topic). When you publish a message, you publish it to a topic, and when you consume a message, you can consume from a topic. Topics are always multi-producer, wit ha topic having zero, one, or more producers that write data to it.
	- ==**Producers**==: Write messages/records to Topics.
	- ==**Consumers**==: Read messages/records off of Topics. Consumers read message from Kafka topics using a **pull-based** model, unlike some messaging systems that *push* data to consumers. Kafka consumers actively [[Poll]] the broker for new messages at intervals that they control. This pull-based system allows consumers to control their own consumption rate, simplifying failure handling and preventing overload of slow consumers. If a consumer were to go down and restart, it reads its last committed offset from Kafka and resumes processing from there, ensuring no messages are missed or duplicated.
		- Note that in a Consumer Group, each Topic Partition is consumed by a single Consumer. Each Topic Partition is guaranteed to be processed in order, but events across partitions for a given topic (i.e. "events in a topic") are NOT guaranteed to be processed in the order that they're published!
		- So every Consumer has 1+ Partitions from a Topic assigned to it (and afaik might even process multiple Topics if you want). If a Consumer goes down, its assigned Topic Partitions will be reassigned by Kafka to the other Consumers in the Cluster. When a Consumer comes back to life, it reads the latest Offsets for its assigned Partitions and continues from there.
	- **==Record:==** A message in Kafka. Each message in a Kafka partition is assigned a unique offset, which is a sequential identifier indicating the message's position in the partition. This is used by consumers to track their progress in reading messages from the topic. As consumers read messages, they maintain their current offset and **periodically** commit this offset back to Kafka, such that they can resume reading from where they left of, in case of failure or restart. Specifically, ==consumers commit their offsets to Kafka after they process a message;== it's the consumer's way of saying "I've processed this message and can move on."

Q: What's the difference between a **Topic** and a **Partition?** 
- A topic is a logical grouping of messages. Ways of organizing your data.
- A partition is a physical grouping of messages; it's a file on disk. Ways of scaling your data.

Importantly, ==You can use Kafka either as a [[Message Queue]] or as a [[Stream]]==.
- The only meaningful difference here is how consumers interact with the data;
	- In a Message queue, consumers read messages from the queue and acknowledge they've processed the message.
	- In a Stream, consumers read messages from the stream, process them, and don't acknowledge that they've processed the message, allowing for more complex processing of the data.

Messages (**Records**) in Kafka Structure has ==four main attributes==:
- ==**Key**== ("Partition Key"): Determine which partition the data lives on. Is optional on an event being written; if not specified, they assign it round-robin to a partition. The common path is that we **do** specify a key.
- ==**Value**==: ...
- **==Timestamp==**: Determines ordering; if none is specified, we use the machine time
- **==Headers==**: Like HTTP threads, it's jut some keys and value that specify information

We can use the Kafka cli or common client libraries in any language that has a client.

Once the Kafka cluster receives a message, ti needs to determine where it should put the message
- Determine what the topic is
- Determine what the broker is
- Determine what the partition is

![[Pasted image 20250521103345.png]]

We hash the key using a [[MurmurHash]], which is a fast hash function, and we modulo that hash over the number of partitions we have. That resulting number corresponds to the partition!

Next, we want to identify which **Broker** (server) that **Partition** exist on. There's a controller in the Kafka cluster that keeps the mapping of **Partitions** to **Brokers**. We look up what **Broker** we want to send it to, and that **Broker** receives the Record and appends it to the append-only log file.

When the **Broker** appends the **Record** to the appropriate **Partition,** it appends that message to its append-only log file for that partition.
![[Pasted image 20250521103635.png]]
So we have msesages A,B,C,D,E... each of which have a correspodning **offset** in the log. ((Idk why he did it "backwards" to me))
- As **Consumers** want to consume a message, they consume a message by specifying the offset of the last message they read!
	- So if they already read Msg A and Msg B, then they know their latest offset was 1, and they'll ask for offset 2 the next time they want to go get some work from the queue.
- **What if a Consumer went down and doesn't know the offset it read?**
	- It will periodically commit its offsets to Kafka itself! So Kafka maintains the latest offset that is read for a given **Consumer**.
![[Pasted image 20250521104055.png]]
- In the case where we have a **Consumer Group**, then Consumer 2 can read and ask Kafka for the next message, and Kafka knows which offsets have been read by the other consumers in the Consumer Group, and it can give you the latest message and commit that offset.... so that everyone knows what the latest offset that's been read is.


For producers, we can create records either from the CLI or via the (e.g. KafkaJS library).

![[Pasted image 20250521104630.png]]
In order to ensure **Durability** and **Availability**, we have a robust [[Replication]] mechanism in Kafka!
- Each partition has a **Leader** replica, which is responsible for handling **==All read and write requests to the partition!==**
- The assignment of the **Leader** is managed centrally by some cluster controller, making sure that each partition's leader replica is distributed effectively across the cluster to balance the load.
- We then have **Followers**, which can reside on the same or different brokers; these don't handle client requests, and just passively backup writes from the leader, and are just ready to take over from the leader if things go down.

So looking at the diagram again:
- We have **Producers** which interface with our Kafka **Cluster** via a **Producer API**
- Our Kafka **Cluster** is made of **Brokers** (just servers)
- On these **Brokers** we have a bunch of **Partitions**, which are just append-only log files
- These **Partitions** are grouped/labeled based on a Topic (e.g. Topic A, Topic B)
- **Consumers** Consume from the **Cluster** via the **Consumer API**, and a **Consumer Group** consumes a specific **Topic** (e.g. Topic B)
	- A Consumer will subscribe to (e.g.) Topic B and read all messages from Topic B.
- **Consumer Groups** make sure that **Consumers** in the group read the same message only once; they don't read the same message twice.
	- The primary purpose of CGs is to allow for parallel processing of data and ensuring that each message is processed exactly once (within the group).
	- When a Topic is divided into Partitions, Events are assigned to partitions based on a partition key. Mesages within a topic-partitoin are ordered, but there's no ordering guarantee across partitions.
	- When a Consumer Group consumes from a Topic
		- **==Each partition is assigned to exactly one consumer within the group!==**, with a Consumer being able to consume from multiple partitions. So no two consumers in the same group can consume from the same partition simultaneously. The Kafka Broker's group coordinator handles this partition assignment to ensure no overlap.


# When to Use Kafka in an Interview

### Use Case: ==Any time you might need a message queue==
	- 	- If processing can be done asynchronously, e.g. **Youtube Transcoding**
![[Pasted image 20250521105758.png]]

- When you upload a Video on Youtube, you upload the full video and store that video in object storage; then you need to Transcode that video, turning it into 360p, 480p, 720p, 1080p; This process takes a while, and can happen asynchronously!
- This message can look like "VideoId": 123, "VideoLink": s3://123...
- Our consumers are our Transcoders, which look at a record in a topic partition, do their transcoding, and store the result in S3.


- ==Any time you need In-Order Message Processing==
	- e.g. **TicketMaster waiting queues**!
![[Pasted image 20250521105855.png]]
- We might want to put people into a waiting queue and let them out of the queue 100 at a time, so that there isn't that much contention for the remaining seats.
- When a user tries to view an event, we put them on our Kafka waiting queue
- Periodically ,every 5 minutes, we pull the next 100 off the queue and let our clients know that it's their turn to actually start to book the event.
- ==Notice that here the same service is both the producer and consumer. That's fine!==


- ==When you want to decouple the consumer and the producer so that they can scale independently!==
	- i.e. **LeetCode** or **Online Judge**
![[Pasted image 20250521110033.png]]
- In LeetCode, a bunch of users in a competition might submit their code, and we need to run their code and give them a response.
- A bunch of users might submit their code at once (maybe 100k users!) and so we need to horizontally scale this primary server accordingly.
- These primary servers are going to put the event on Kafka.
	- We don't want to have to scale these servers that are doing the running of the code, perhaps because we're cost sensitive!
	- So we can just scale our cheaper Primary Servers, which put the events on the Kafka queue, and later the (smaller, not scaled) workers will get the events from the queue and process them; the workers don't have to scale.


## Use Case: ==Any Time You Need a Stream==
- A Stream is used when you need to process a lot of data in real time, like in an **Ad Clicking Aggregator system**
- ![[Pasted image 20250521110506.png]]
	- In a Queue, the consumer gets to choose when it wants to read off the queue, and things can exist in the queue for a long time
	- In a Stream, you have an infinite stream of data coming onto Kafka, and you want to be consuming it as quickly as possible in order to get some real-time statistics or updates.
- When a user clicks an Ad, we put that click on our Kafka queue, and then we have a Consumer, in this case [[Flink]], reading off that stream in real time and aggregating those so that we can tell our advertisers how many their ad has been clicked.


- Another use case if if you have a **Stream of messages that need to be processed by multiple consumers simultaneously, like in Messenger or FB Live Comments**
- ![[Pasted image 20250521110554.png]]
- Here, we have a stream of messages being processed by multiple consumers simultaneously -- also known as [[Publish-Subscribe|Pub Sub]]!
- If we have a Commenter that leaves a comment on a live video, we can put that comment on Kafka as a pub sub, and have all of these other servers connected to the users that are watching the live video... and they're subscribed to the stream, and if they see a new comment come in on a live video that they know their client is subscribed to, they read the event and send the information to the user ((I assume either via [[Polling]] or via [[Server-Sent Event]]s)), so that it looks like the message showed up in real time for the user! (e.g. the comment appears and floats down the video, or whatever).




# What to know about Kafka for Deep Dives in Interviews
Things to know include:
1. **Scalability**: How do you scale the system, and how do you scale Kafka?
2. **Fault Tolerance and Durability**
3. **Errors and Retries**: What happens when things go wrong?
4. **Performance Optimizations**: Specifically for real-time use cases where you have a lot of events and you need to make sure throughput is as high as possible
5. **Retention Policies**: To minimize the amount of storage on disk.

#### Dive: Deep Dive Scalability:
- "How is this going to Scale?" (usually in the context of the full system)
- **Constraints work knowing**:
	- There's ==no limit on the maximum size of a message in Kafka, but it's highly advised to **keep these messages < 1MB!**==
		- **==COMMON MISTAKE==**: Candidates putting the entire media blob on Kafka!
			- e.g. if we're doing Youtube transcoding of videos into different resolutions...
				- GOOD: Storing the video blob in object storage and putting the S3 URL for the blob into the event. The consumer takes the event, finds the url, downloads the video via the S3 url. The only thing on Kafka is the videoId and the s3Url.
				- BAD: Putting the ENTIRE VIDEO in the Kafka event! Yikes!
	- A single ==broker== can store ==~1TB of data== and can handle ==~10k messags per second== (handwavey; depends on the hardware)
		- In the interview, do the math! Does your system exceed this? If it doesn't we might not worry about scaling the number of brokers at all!
- **In the case where we DO need to scale, what should we do?**
	- Introduce more Brokers (servers; More memory, more disk, can store and process more messages)
	- Choose a good partition key!
		- Ideally, we'd like to uniformly partition data across our brokers, right? We don't want a ==hot partition== that gets overwhelmed and others getting no data!
		- Complication: At the same time, if we need some in-order processing of events, note that order is only guaranteed within a single topic partition!
	- Let's talk about **==how to handle hot partitions==**
		- ((Aside: Nowadays, many of these scaling considerations these days are made easy by managed services from cloud providers, like Confluent Cloud or [[Amazon MSK]], which handle a lot of these scaling things for you; you still need to pick partition keys though))
		- Recall: a ==Hot Partition== is one where ~everything is going to a single partition/broker, and it's overwhelmed.
			- Consider an **Ad Click Aggregator**: We have a stream of data, where User A clicked on Ad B; We take that stream of ClickEvents and aggregate them to understand the metrics of how often ads are being clicked on!
			- Naturally, we might partition by **AdId,** right? But if Nike ads a hot new add that everyone's clicking on, that partition might get quite hot! So ==how do we handle that hot partition==? 
		- Options
			- **==Remove the Key==** from your messages 🤷‍♂ such that you randomly distribute your messages across partitions, for this topic. ==If you don't care about ordering at all, this is fine and works!== It seems simple, but if you don't need ordering, it's a great option.
			- Add a **==Compound Key==**; Instead of making the key **AdId**, we can make it **AdId:UserId**, or a ==common pattern== to do is to take our **AdId** and partition it across 10 partitions by doing **AdId:1-10**, adding a random number 1-10. This makes sure that our Nike ad is split across 10 different partitions, reducing the strain on the single hot partition.
				- You'll have some logic in the producer which understands which partitions are hot and does this concatenation for those keys. For the Consumer, there will now not be ordering across those (e.g. 10) partitions.
			- Apply **==[[Backpressure]]==**; A producer understands that a given topic partition is overwhelmed at the moment, and slows down its rate on production. This **might work** for your service.

Aside: From the Article.... ==There are a few strategies to handle hot partitions:==
1. Random partitioning with **no key**:  If you don't provide a key, Kafka will randomly assign a partition to the message, guaranteeing even distribution. The downside is that you lose the ability to guarantee ordering of messages.
2. **Random [[Salt]]ing:** Add a random number (e.g. 1-10) or a timestamp to the ad ID when generating the partition key; may complicate aggregation logic later on the consumer side.
3. Using a **Compound Key**: Instead of using just the ad ID, use a combination of ad ID and another attribute, like a geographical region or userID segments, to form a compound key. (Seems pretty similar to Random Salting, but perhaps more purposeful)
4. [[Backpressure]]: Slow down the producer! A managed Kafka service may have built-in mechanisms to handle this, or you can implement it yourself by having the producer check the lag on the partition and slow down if it's too high.

### Dive: Fault Tolerance and Durability
- One reason you might have chosen Kafka is that it **has really strong guarantees on durability!**
	- For each **Topic Partition**, we have a **leader replica** and a number of **follower replicas** that are ready to take over if the leader goes down. ==A replication factor of 3 is common== (meaning a leader and two followers)
	- The things that are more important is that there are **Two Relevant Settings**:
		- When you set up a Kafka cluster, you have a config file. Two of the settings are:
			- ==acks==: How many followers need to acknowledge a write before we can keep going? **acks=all** means maximum durability, where every follower needs to acknowledge that they also got the message, then we can say that we also got this message.
				- The tradeoff is **durability versus performance/speed**. If acks is low, it's possible that we could lose data!
			- ==replication factor==: How many followers we're going to have. Default is 3. Again a **durability versus performance/speed/space** option.
- **==What happens when a Consumer goes down?==**
	- **==NOTE:==** Kafka is often thought of as **always available;** if an interviewer asks you "What happens if Kafka goes down," this is often isn't very realistic; Kafka doesn't go down because the durability guarantees that we just discussed.
	- A more realistic question is what happens when a consumer goes down.
- ![[Pasted image 20250521115527.png|600]]
	- Recall: We have the cluster, and our consumer firsts reads a message, then gets the offset of the message it just processed, and commits the **offset** of the message back to Kafka, saying "I've processed X".
	- If we have a Consumer Group, then each Consumer is responsible for a **range of Topic Partitions!**
		- ==So if a Consumer in a Consumer Group goes down, we need to do some rebalancing. That's all going to be handled by the Kafka Cluster, which will update the existing consumers on their new ranges.==
	- It's important to know when to commit your offset, which is an action that says **"I've finished teh task that I've been asked to do.**
		- For a Web Crawler,  we don't commit the offset until we have confirmation that the HTML that we've downloaded is stored in S3. If we commit the offset BEFORE we know the data is stored in S3, then if our consumer goes down... it's possible that we effectively skip an event.
- **Let's talk about Errors and Retries**
	- Our system may fail getting messages into or out of Kafka! Producer Retries and Consumer [[Retry|Retries]].
	- For ==Producer Retries==: The Kafka Producer API supports some configurations that let us retry gracefully.
		- ![[Pasted image 20250521115903.png]]
		- You'll want to be sure that you've enabled [[Idempotent]] producer mode, which is to avoid duplicate message when retries are enabled; it makes sure that messages are only added once if we accidentally send it twice.
- What about ==Consumer Retries?== **==NOTE: THIS IS MUCH MORE INTERESTING IN KAFKA!==**
	- ![[Pasted image 20250521121747.png]]
	- In a Web Crawling Example: Our Kafka broker is storing URLs of websites we need to crawl.
		- Our Consumer pulls these URLs and then goes and fetches them from the internet in order to get the HTML and store it in some blob storage like S3.
		- The Web is a dirty place, so maybe when we pull a url from the queue, and that ==**website is currently down**! What do we do, as as Consumer?==
			- Kafka does actually not support Consumer Retries out of the box! Interestingly, [[Amazon SQS]] *does support Consumer Retries!*
			- In Kafka, what we do is: After N failure to try to get the site, our Consumer will place a new message on a ==Retry Topic==, where that message says the number of retries... and some consumer (perhaps in a different CG) retries it.
				- If that number of retries exceeds some number, say 5, then we can say that that message is permanently failed, and we can put it into a ==[[Dead Letter Queue]] Topic==. This just stays here and no one reads from it; an Engineer can later go and manually look at these.

From the Article.... ==When a Consumer goes down, what happens?==
- Kafka has fault tolerance mechanisms to help ensure continuity:
	- **Offset Management**: Because partitions are just append-only logs wher each message is assigned unique offset... Consumers commit their offsets to Kafka after they process a message. This is the consumer's way of saying "I've processed this message." If a Consumer diesa nd comes back, it reads its latest offset from Kafka and resumes processing from there.
	- **Rebalancing**: When part of a consumer group goes down, Kafka will redistribute the partitions among the remaining consumers so that all partitions are still being processed.

### Dive: Performance Optimizations
- We want to be mindful of the performance/throughput so that we can process messages as quickly as possible!
- Producer Side:
	- Batch messages on the Producer (this is a configuration that we can set on the Kafka Producer)
	- Compress messages in the Producer (this compression is supported via the Producer clients; we can GZIP our messages so that they're smaller.)
	- By ==Batching== and ==Compressing== our messages, we improve throughput by sending fewer, smalle messages.
- Broker Side:
	- The biggest impact you can have on performance is the choice of your **Partition Key**: The goal is to maximize parallelism by making sure that our messages are evenly split among our partitions and our brokers.
		- ==You should always start with talking about the Partition Key that you're going to use in Kafka for a Topic, it's incredibly important==

## Dive: Retention Policy
- There are two settings in Kafka's configuration that determine how long we keep a message around before purging it
- Kafka **Topics** have retention policies that determine how long messages are retained, via two settings, primarily:
	- retention.ms (default 7 days): How long to keep a message
	- retention.bytes (default 16GB): When to start purging, based on size
	- (==Whichever of these come first is when purging starts==)
- You can imagine a system where you have to be able to replay events months later; you can imagine that you have to discuss this in an interview, configurng the retention policy to keep these messages for a longer duration, but ==youll want to talk about the impact that this will have on storage costs and performance.==






------------

----------



# Motivating Examples (Intro)
![[Pasted image 20250520195007.png]]
Imagine we're running a website where we're presenting **real-time event updates on each of the games in the World Cup!**
- Events, when they occur, are placed in a queue!
	- We call the server or process that's responsible for placing these events on a queue the ==**Producer**.== It might be someone with a laptop sitting at each of the game!
	- At the other end of the queue, we have a process/server that reads events off the queue and updates our website. We call this our **==Consumer==**.
- Our events are being processed in order, our site is getting updated, etc. and people are loving our project!
- But then FIFA decides that they want to expand to 1,000 teams, and they want all events to go at the same time.
- Perhaps it's the case that our single server is struggling to keep up (this isn't realistic, but go with it).

Problem 1: Too many events on the queue, and the server that's hosting the queue has run out of space!
Solution 1: Let's [[Horizontal Scaling|Horizontally Scale]] our queue and randomly write to one of the nodes! 

![[Pasted image 20250520195337.png]]
- But this introduces a new problem! Now we have two or more additional servers... but as a producer produces an event, if we're randomly distributing events across the server, we'll end up with a problem: The consumer might read that goals are scored before it sees that the match is even started, etc!
	- This is because the consumer is just reading off of both of these queues, and there's no way to maintain order any more.

To solve **this** problem, we should ==distribute the items into a queue based on the game it's associated with!==
![[Pasted image 20250520195521.png]]
Now because these Brazil/Argentina games are all on the same queue, we can guarantee that events for a specific game are procsesed in order!

**Fundamental Idea:** In order to scale, messages sent and received through Kafka require a ==user-specified distribution strategy!== The user needs to specify how we distribute our events across our different [[Partition]]s.

There's a third problem though: We've scaled to so many events that our **single consumer can no longer keep up!**
![[Pasted image 20250520201357.png]]So let's add some more consumers, right?
- But what happens if Consumer 1 and Consumer 2 both read the Argentina score event and increment the score twice? oops! Enter Consumer Groups.

With **==Consumer Groups==**, each event is guaranteed to only be processed by one consumer in the group. We can group these Consumers into a Kafka consumer group, and make sure that each event is only processed by one of these consumers in the group.

And what if FIFA decided they wanted to expand their hypothetical world cup to even more sports, like Basketball?
- But we don't want our Softball website to cover Basketball events, and vice versa.
- So we introduce a new concept called **==Topics==**.
![[Pasted image 20250520201600.png]]
So now every Event is associated with a Topic, and Consumer Groups subscribe to a specific topic.
- See that we have a specific Consumer Group for each Topic, and each Topic can still be still partitioned (and later, will be replicated as well) among multiple (virtual) servers.





