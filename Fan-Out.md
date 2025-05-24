

The idea is that rather than do a really expensive read query, we want to **do more work on write path** to deliver the data to users that are interested, so that we can **do less work on the read path.**
- Push notifications
- Social media newsfeed
- Mutual friend lists
- Stock price delivery (who are the interested parties when the price of Google goes up or down by a dollar)


Solutions:
- Synchronous delivery to every interested party (naive, request will time out)
- Asynchronous delivery via [[Stream Processing]], e.g. using Flink
	- Put message into log-based message broker; consumer receives it, figures out where message needs to be sent, and does so accordingly. 

![[Pasted image 20250524110029.png]]
(So I guess in this one the won't time out because it's happening offline with respect to the request, but still a Consumer has to fan out a tweet to 200,000 twitter followers (This is the [[Hot Spot]] problem for fanout, I think). I could imagine that a Consumer wouldn't want to make 200k writes to different Redis queues, no?)


One problem we'll often have is the **Popular Message** problem: If too many users are interested in the message, it becomes very expensive to send it to all of them!
- Hybrid Approach: Instead, the Stream Consumer will notice how many places it has to deliver to, and instead opt to deliver it to some sort of "Popular Message Cache"
	- So tweets from non-popular accounts are directly inserted into their followers message caches...
	- Tweets from popular accounts are pulled from the popular cache that users can occasionally poll from. 

