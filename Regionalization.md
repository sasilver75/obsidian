

In a SD interview, you'll be asked to design a system that works across the whole globe. When we talk about networking across the globe, we have to deal with the las of physics! The speed of light can only travel so fast in a fiber optic cable from London to New York. I'll have at least 80ms of latency, no matter what I'm doing! That can be a problem, if we're making lots of requests.

In global scaled systems, we often have global-scaled traffic!
- If we're designing Uber, which operates in many locations, we can sometimes simplify problems by recognizing where there are neat partitions in the problem.
	- For Uber, we can recognize that riders are typically only requesting drivers in cities that they're already in! I don't request a London-based driver from Seattle, for instance.
	- This simplifies the problem! We can take our system design and duplicate it in the relevant regions that we're doing business in; these operate at a much smaller scale, and our users get better latency too!

Theres' a couple of things to keep in mind:
- We want to keep the core of our data and processing as close together as we can.
- If we have requests coming in from our users from London to New York, we're probably going to make many requests back and forth between our webserver and database to figure out what the response is... it would be very silly if our webserver were in London, and our database were in New York.

Some problems:
- I can't always depend on moving my data around the world! In the Reddit case, I have users in Japan who are accessing posts created by people in the UK that are interacted/commented on by people in the US. In this case, I need my data to be everwhere!
- You have some options:
	- Have data local to where it's *most likely to be used.*
	- We can replicate our data; When we write to a database, we store it locally, and make sure that when it gets replicated, it gets replicated to other regions so that it can be read.
		- Assuming [[Asynchronous Replication]], this implies [[Eventual Consistency]], which is probably ok.
	- Or, we can introduce a [[Content Delivery Network|CDN]] for some use cases. We have an [[Origin Server]] that pushes out content to [[Edge Server]]s. When someone requests the front page of Reddit, and it's already sitting on a CDN, I can return that to users pretty immediately.







