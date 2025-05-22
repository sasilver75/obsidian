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
==This redistribution is **really bad!**== It creates a surge in database activity, and can slow down or even crash our site!
- We only want to move the data that we have to!


[[Consistent Hashing]] is the solution here, consisting of three steps:
1. Create a Hash Ring with a fixed number of points (e.g. 0-100; though in reality, it's 0-2^32)
2. We evenly distribute our databases around this Hash Ring
3. To know which database an event should be stored on, we hash the EventId, find that point on the ring, and then walk clockwise until we hit a database!
![[Pasted image 20250519175828.png]]

So if we want to add a fifth database, which let's say just hashes to 90 on the ring, then the only events that need to be redistributed are those in the 75->90 range, which previously were assigned to DB1:
![[Pasted image 20250519175916.png]]

And the same thing would happen if we were to remove a database!
![[Pasted image 20250519175935.png]]
Only events that hashed to a spot between 0-25 need to be moved (and they need to be moved to database 3)

But there's a little problem, with that right? In this picture, we're saying that Database 3 now has 2x the data of DB1 and DB4! So we'd like to make sure that the data is more evenly distributed when a database exits the ring...

The solution is something called **VIRTUAL NODES**!
- Instead of putting every database at one position in the ring, we put it at multiple!
![[Pasted image 20250519180105.png]]
Now if database 2 is removed, instead of all of the events in the range 0-25 having to be remapped to DB3, you see that we have a much more uniform distribution of the events.


## So when does Consistent Hashing come up in a System Design Interview?
It's used inside many of your favorite databases, like: Redis, Cassandra, DynamoDB, many CDNs
- In an interview, you might make a nod to this, but the only time that you really need to go deep into designing an algorithm is when you're designing a single scaled backend component ("design a distributed cache," "design a distributed message queue," etc.)














