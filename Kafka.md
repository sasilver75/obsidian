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
	- **==Broker==**: A server (physical or virtual) that hold the "queue."
	- **==Partition==**: The "queue". An ordered, immutable sequence of messages that we append to, like a log file. Each broker can host multiple partitions. A *physical* grouping of messages.
	- ==**Topic**==: A logical grouping of partitions. You can publish to and consume from Topics in Kafka. A *logical* grouping of messages (across physical/virtual servers, each hosting a partition for a given topic).
	- ==**Producers**==: Write messages/records to Topics.
	- ==**Consumers**==: Read messages/records off of Topics.

Let's talk about what happens in a lifecycle of a message through a Kafka cluster.
- 

# Motivating Examples
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









# Overview
- 


# When to use it in an Interview
- 


# What to know for Deep Dives
- 



