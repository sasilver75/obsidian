SDIAH Link: https://www.hellointerview.com/learn/system-design/deep-dives/consistent-hashing 

Consistent Hashing is easiest learned via an Example:
- Imagine we have a simple events website, like TicketMaster.
- It starts with a single server, and a single database storing our event information.
- As the site becomes larger, we have too many events to store on a single database, so we add additional databases (totaling 3) -- but there's a problem: Given a certain event, how do we know which of our Databases holds the relevant event information?

One approach is to hash the EventId with some sort of hash function (e.g. [[MD5]], [[MurmurHash]], etc.), which gives us a large number, which we then modulus by the number of databases we have to get the "index" of the database the event should live on.

But what happens when we then add a Database 4?
- Uh oh, now we've changed the **modulus** that we were doing before! 
		- Before, we were doing `hash(eventId) % 3`, but now it's `hash(eventId) % 4` -- this is going to mean that we have to do a LOT of data redistributing! 
		- Ideally, ==we'd prefer to only redistribute the data that is going to go to our database 4, rather than ~all the data!==

![[Pasted image 20250519174021.png]]
This redistribution is **really bad!** It creates a surge in database activity, and can slow down or even crash our site!

We need to  






