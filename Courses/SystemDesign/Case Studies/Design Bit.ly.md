SDIAH Video: https://youtu.be/iUU4O1sWtJA?si=rt3ppga7-Zpqrx25
SDIAH Writeup: https://www.hellointerview.com/learn/system-design/problem-breakdowns/bitly

Everyone's first system design question is "Design a URL Shortener."

Framewokr:
![[Pasted image 20250523113102.png]]
- Requirements:
	- Functional Requirements: The core features of the system; things the system has to be able to do. "A user can..."
		- Hopefully this is 2-3 features.
	- Non-Functional Requirements: The "-ilities" of the system. Scalability, monitorability, durability, latency, etc.
- Core Entities
	- High-level list of the core entities that are persisted in our system. They "Tables"
- API or Interface
	- Contract between your users and your backend services.
- Data Flow
	- Only applicable to infrastructure questions like design a distributed cache, message queue, etc.
- High-Level Design
	- Drawing boxes and arrows to outline the highlevel system that satisfies the Functional Requirements.
	- This system isn't going to scale or be able to do anything fancy, it's just going
- Deep Dives
	- Going one by one through non-functinoal requiremnts to align our high level design in a way such that it satisfies our functional requirements.


----------------

Let's start 

# Functional Requirements
- If you're designing a system you've never heard of, this is the time where you'd pepper an interviewer with questions about the service you're trying to build.
	- In our case, a **URL Shortener** is a service that converts long URLs into short URLs, and those short URLs will redirect us to the long URL when accessed.
- Requirements List
	- ==Users can create a Short URL from a Long URL==
		- Users can optionally specify a ==custom alias==, so that users can come to us with their own short alias (e.g. "evan", so the URL is bit.ly/evan or whatever.)
		- Users can optionally specify an ==expiration time== (e.g. if they're making a ShortURL for a conference or something. If sometime tries to navigate to that ShortURL after the expiration, they'll get an error of some sort.)
	- ==Users can be redirected from a ShortURL to a LongURL==

# Non-Functional Requirements
- These are the qualities of the system! These -ilities: scalability, latency, durability, fault tolerance, compliance, etc.
	- This is where you talk about the [[CAP Theorem]]
	- What qualities does the system have to have to be an app that lives up to user expectations?
- Maybe you go one by one through these -ilities, having a list on your table during the interview!
- Think about which of them are interesting and relevant to the system.
- For example, we might consider **Latency** first:
	- It's important that the redirection is low-latency!
	- ==**The system should be low-latency on redirects**.== If we give it a ShortURL, we should quickly go to that URL.
		- (It's not just "low latency," it's "low-latency on redirects.)
		- We should even try to quantify it, something like **"less than 200ms"** (which is a good rule of thumb for 'perceived as realtime by users')
- We might consider the **Scale** of the system
	- "How many users are we designing for? What's the scale of the system?"
	- "We can expect 100M DAU, and maybe 100B will get shortened in totality, over all time."
	- ==**NFR: "The system should be able to scale to support 100M DAU and 100B URLs in totality"**==
- Another requirement: The Short Codes need to be unique so that there are no collisions, so that people don't get redirected to a site that they didn't expect to go to!
	- ==**NFR: The ShortURL uniqueness should be enforced==.**
- Considering the **CAP Theorem**:
	- In distributed systems, in the face of network partitions, we have to decide whether we want our system to have **strong consistency** or **high availability**.
	- In an interview, you should think: **Do I need strong "Read After Write" consistency? Does everyone ned to see all writes that have been written?**
		- If you're doing a [[Design TicketMaster]] system, then yeah, maybe! We don't want two people getting 200s on buying the same ticket!
	- In the case of our URL shortener, do we need that strong read-after-write consistentcy?
		- Imagine someone gives us a LongURL, we shorten it, and give back a ShortURL
		- Does anyone who then uses that ShortURL immediately **NEED** to be able to see the site immediately?
			- No, I don't think so. If they didn't get redirected in the first 30-60 seconds, who cares? We can give them a little error saying "That's not available yet!" or something like that.
	- No given that ==we don't need [[Strong Consistency]], so we can settle for [[Eventual Consistency]] for our URL Shortening!==

##### Back-of-the-Envelope Calculations
- They advise aginst doing up-front BOTE calculations; they'll do some basic math about storage or latency and then look at the interviewer and say "Okay, it's going to be a big distributed system."
	- That's no news to anyone! **You're likely not going to change your design based on this information**
- Say: =="I'd rather not do math up front, I'm going to do math as I need it during my deep dives if I NEED it to make a decision."==
	- ==We're only going to do calculations when the result would directly inform your design==. Doing it up front doesn't satisfy that.


# Core Entities
- These are the "tables." At this point, we don't even want to detail the fields and columns for these.
- We have:
	- LongURL
	- ShortURL
	- User that creates them
- At this point, the reason we don't outline it is that for more complicated systems, this might be too early to really understand our full data model.


# API Contracts
- The way we suggest to go through this is very methodically: **==Go to your functional requirements, and for MOST cases, it's a 1:1 from functional requirements to APIs that you need to provide==** (sometimes it's >1 for each FR).
	- Just say: "Okay, here's a functional requirement. Let's make sure that we have a URL that supports that."

So we might think:
- (==Note that I'm not even specifying the datatypes for these; it's a trivial detail. Note that we're using **?** for the optional bits.==)
- Note that we're using REST with GET/POST/PATCH/PUT/DELETE as appropriate.
- Note that for our paths, we're going for a Plural Noun; sometimes you'll have a REST zealot at the lower level, someone who really cares about this.
```
%% FR: User should be able to Shorten a URL %%
%% Shorten a URL %%
POST /urls -> shortURL
{
shortURL
customAlias?
expirationTime?
}

%% FR: User should be able to be redirected %%
%% Redirect %%
GET /{shortUrl} -> Redirect to Original URL
%% ((To be clear, here, we're hitting OUR SERVICE with the ShortURL, e.g. bit.ly/samsURL, and then we're getting redirected to canoeclub.com)) %%
```

So that's all we need for this problem! We've satisfied all of our functional requirements.

# High-Level Design
- Our primary goal here is to just satisfy these core functional requirements
	- We're not going to worry about Scale, Scalability, Availbility Ensuring URL uniqueness, or any of these things; we just want a simple design down first, letting our mind work linearly, so that if we're given a design we haven't seen before, we can build up a design step by step.
- Tip: **==Look at your API contracts, and go through each of them, making sure that you have a high level design that satisfies each. Make sure you go through the input and output for each request, and how the data needs to get trasnformed as it moves through your high-level system==**

- Boxes:
	- We have a Client
	- We have a Primary Server
	- We have a Database
The client makes a request to your backend server, the server does something, persists it to the backend database, and returns information to the client.

![[Pasted image 20250523120748.png]]

Let's look at our APIs:
- POST /urls
	- Client is going to make an API request of getShortURL(long, alias?, expirationTime?) to hit our Primary Server, which will generate that ShortURL, save it to the Database, and then return it back to the client.
			- **(For now, we've blackboxing the ShortURL creation process, that's fine! "I'm blackboxing this for now, and will come back to it later.)**
		- But what is it saving to the database? We can now define our datamodels!
		- Because ShortURL and LongURL are so simple, let's have a single URL table that has columns:
			- shortURL/customAlias
			- longURL
			- creationTime
			- expirationTime?
			- userId FD to User
		- And a User table that has 
			- userId
			- ...
			- ==**(There will likely be other columns in a user table; email, passsword hash, salt, all of these different things: WHO CARES! Don't get distracted by this, focus on the core functionality)**==

- GET /{shortURL}
	- Client is going to make a API request of redirect(short), and the Primary server will lookup the long from the short and redirect the client
	- Primary server will respond with a **302 REDIRECT** status code, which says: "Take this URL and automatically navigate to that URL"
		- Note that **301 REDIRECT** is also an option, but a 301 REDIRECT is a "**permanent redirect ("moved permanently")**", but it means that a client can cache the redirect, and in the future might not even go to your service, and just redirect immediately.
		- In the 302 case, this is temporary. They're always going to come to us with the ShortURL, we'll then redirect them to the LongURL
		- It depends on your requirements
			- 302 is good if you want analytics; we can then show users how often the ShortURL is being queried, since the request is always coming to our server. Even if we aren't showing users analytics, you'd be tracking that internally as a platform/product, and if you saw things fall off a cliff, you'd know that something broke. And if you went for a 301, you'd lose this visibility, so you should probably 
			- In the 301 case, we aren't going to get the opportunity to do that. But it means that we might need fewer servers.
	- We also need to check if the ShortURL we retrieved has **expired**, if that's been specified.


Great, so we've accomplished all of our functional requirements! (**==Double check that you have==**)



# Deep Dive: Let's expand the design to support our NFRs!
- Our NFRs were:
	- Low latency onredirects (~200ms)
	- Scale to support 100M DAU and 1B urls
	- Ensure uniqueness of short code
	- High availability, eventually consistency for URL shortness

### NFR: Ensure Uniqueness of Short Code
- Let's start with **==Ensure uniqueness of short code==**, which we had black boxed earlier (it's the hard part, and interesting part of a URL shortener, and we just handwaved it earlier in the HLD so that we didn't get bogged down before satisfying our FRs)
	- So...what do we need?
		- We need the creation to be "fast"
		- We need the creation of the ShortURL to be unique
		- We need the ShortURL to be... short (how short? We can ask our interviewer, or guess. Typically something in the 5-7 character range.)
	- Let's talk about what options we might have, building from the most solution to something more complex
		1. We could just **take a prefix of the LongURL**, right? Just the first 5-7 characters, and use that as our ShortURL
			1. This is **obviously bad**, because tons of urls (e.g. all those starting with www.twitter.com/posts/)... so we'd have a lot of collisions.
		2. We could use a **random number generator
			- How large should/would this number need to be?
				- Let's recall that we're going to have 1B URLs, so we need to have AT LEAST 1B. 1B is 10^9 , or 10 characters. That's more than our 5-7 -- that's too many characters! We'll need to do something more sophisticated to compact this, right? And we'd probably need more like 14 or so, to make sure that the collision rate is lower...
				- But we could use [[Base62]] encoding, which incorporates the alphabet too!
					- Base62 uses: 0-9, A-Z, a-z, so that it can encode 62 options per character!
						- So we can take a large number, then Base62 encode it.
						- If we had a code that was length 6, then it would be 62^6 possible combinations, which is ~56Billion. That's pretty good! We have a lot of space there.
				- So what's the probably that a random number between 0 and 56B... when calculate 1B times, ends up colliding? 
					- The collision probability is still very high! This is something called ==the **Birthday Paradox**, which says there's a 50% chance that two people, in group of only 23 people, have the same birthday!== Despite there being 365 days in a year.
					- We can use the same formula to calculate the probability of a collision on 56B "days", given 1B "students"... it turns out that for 1B random generations, there's an estimated 880K collisions that might happen.
						- That's "a lot" in a sense... how can we address this?
			- We can take this approach of random number generator, Base64 encode, then come to the database and read the ShortURL, see if any exists like this already, and only if any doesn't exist, we then write to it
				- So this just introduced another read to our process...
				- "We just need to check for collisions first." There's an extra read, but that's not the end of the world.
		3. We could **Hash** the long URL using something like [[MD5]] or something cheaper like [[MurmurHash]], to get some output... and you take that Hash and Base62 encode that hash, and slice it to just take the first 6 characters.
			- You end up with the first 6 characters, Base62 encoded... So you end up with literally the same thing we had with the random number generator, same chances of collisions, etc.
		4. Can we avoid checking the database? We can use a **Counter**! Why introduce this randomness in the first place? Why not have the first person have ShortCode 1, the second person has ShortCode 2, etc.
			- So we'll increment a counter, and then again [[Base62]] encode it, to make it more compact. This still lets us get to that **56B**, but we'll never have a collision, because we're just linearly increasing; there's a sequential nature to it. Remember that Base62 doesn't have collisions; it's not a hash function, it's an **encoding**. 
			- **Let's use this!**

![[Pasted image 20250523122039.png]]
- So we'll add a Counter to the PrimaryServer. This guarantees that it's unique, and the Base62 encoding guarantees that it's short (and URL safe)
	- **It's not without its flaws though! It's predictable, meaning tha anyone who wants to know our ShortURLs... can!**
		- If you can count and Base64 encode... then a competitor can:
			- Know how many URLs we've encoded
		- They can scrape all of the LongURLs we have by calling the ShortURL with the incremented code each time
	- **Maybe this is a problem, and maybe it's not!** It's a product decision
		- We can display to the user: "Warning, don't shorten Private URLs, we aren't responsible for this"
		- We can have rate limiting to dissuade competitors from scraping
- But we can also just not make this happen in the first place by introducing something called a **==Bijective Function==** (you don't need to know this)
	- ((Note that Base62 is also Bijective))
	- These are functions that issue a 1:1 mapping. One of the most popular libraries are sqids.org, which take a number and return a base62 encoded string that looks exactly like a bitly one... so it takes the number you provide (counter), and 1:1 maps it to a base62 number in a way where you can't tell what's up in you're a hacker, etc. 
- So ==these bijective functions exist, and it's worth maybe looking up==
- The nice result of the Counter approach is that **we don't need to do that extra "read" step against the database; we know that every single one of the ShortURLs that we create is going to be unique.**
	- So a user provides a LongURL
	- We increment the counter
	- We turn that counter into a Base62 encoding of length 6 (perhaps using either a raw Base62 encoder, or better by using a fancy library does uses a obfuscatory (but still bijective) function and then Base62 encodes the result)
		- ((Wait, the encoding won't be any particular length, e.g. if the counter is 0, the code will just be like.. "n' or something. It's an encoding, so it gets longer as the data you're encoding gets longer.))
	- We write that to the database, confident that no other record in the database will have the same 

### NFR: "Make the redirect as low latency as possible"
Let's move to our next non-functional requirement: Making the redirect as low latency as possible.
- What's happening right now?
	- The user is requesting a shortURL, our primary service is taking that shortURL and looking up the corersponding LongURL in our database, and then redirecting the client to the LongURL.
	- The thing that takes a long time here is going to be the "looking it up in the database"
- So how can we speed this up?
	- Right now, we might have 1B rows in our database, and we might have to scan over all 1B rows to find the URL that we're targeting!
	- So we need to introduce [[Index|Indexing]] to our database, which we haven't mentioned yet.
	- In a Database, we can define a [[Primary Key]], which enforces uniqueness and enforces that an [[Index]] is automatically built on that column in a table.
	- You can think of this Index as something typically kept in memory, which you can think of as functioning as a **pointer to a location on disk.** We can then go exactly to that place in disk, find the LongURL and return it.
- If we were using [[PostgreSQL|Postgres]], then it likely uses a [[B-Tree]] as an index, which is a self-balancing tree. If you're been studying for coding interviews, you know plenty about these trees!
	- This makes the seeking to the appropriate location **Log(N)**.
- You also have the option of creating another [[Hash Index]] on this ShortURL, which would be **O(1)**, which would hash the ShortURL and point directly to the place where our LongURL is. **But realistically you don't need to do this, BTrees are so optimized that this would effectively be the same.**
	- So we're going to memory, using the index to find where to go on disk, and then return the information.
		- **==These days, SSDs are really fast, and so this probably isn't a problem; but if wanted to improve it further, we could add an in-memory Cache like [[Redis]].==**
	- This Cache is just a separate computer whose memory we utilize.
- We're going to make this a [[Redis]] cache in front of our database.
	- This is going to be a [[Read-Through Cache]]: In a ReadThrough cache, if the cache has a miss, the cache automatically fetches the data from the backing datastore, stores the fetched data in the cache, and returns the result to the requesting client.
		- ((This is to compare to a [[Cache Aside]] strategy, though in the picture he's actually doing a Cache Aside strategy, lol))
	- We're going to use [[Least Recently Used]] policy as as [[Cache Eviction Strategy]]
	- These caches are commonly Key:Value pairs.
		- The Key: ShortURL
		- The Value: LongURL

- **We could also cache in a [[Content Distribution Network|CDN]]**, which are little edge services around the world. Say you're in CA, and trying to request a URL from China. There's a lot of latency involved there! If a user in the past in CA has hit a server in China, we might have cached that information in California, which is much closer to our users.
	- ==There's a problem with CDNs for this problem, though; It's the same problem we had considering those 302 permanent redirects; requests will never hit our main server, and so we might lose observability!==
		- ((I'm not sure whether modern CDNs have the ability to contribute to your observability platform, etc.))



### DD for NFR: "Scale to Ensure 100M DAU and 1B URLS"
- We said before that we have 100M DAUs, let's say that they each do... 1 redirect.
- 100M is the same as 10^8; when you're doing ==**Math in a SDE, we suggst you use exponents like this, it makes math easier**==!
	- There's 10^8 users, and ~10^5 seconds in a day; to divide these, we just subtract: 10^3, so ==**1000 RPS== if it's evenly distributed**, but maybe there will be some peaks, so let's x by 10-100x, so let's say ==**10k-100k requests per second at peak==.**

So you need to build up intuition about what "a lot is":
- ==An average EC2 instance, a T3 Medium, can handle ~1000 concurrent requests at a time==. This is intentionally handwavey, depending on how computationally expensive the requests are; how much memory/cpu/bandwidth are being used.
- So if we're using average servers, we can't handle this with one average EC2 server! So we have options:
	- [[Vertical Scaling|Vertically Scale]]: Bigger instance, let's use an instance with more CPU/Memory/Bandwidth -- it's more expensive.
	- [[Horizontal Scaling|Horizontally Scale]]: We have MORE of these commodity boxes, and each request is routed to one of them.
- We're going to choose to scale horizontally here.
- NOTE: There are a lot of reads in our system, but likely fewer getShortURL requests... maybe it's 1/1000th. ==So our write ability doesn't need to scale a lot, but our read ability needs to scale a lot!==
	- So we can evolve our design to a Microservices architecture so that we can scale separately!
		- We'll have a ==Read Service== (handles redirects of ShortURLs to LongURLs) and a ==Write Service== (handles creation of ShortURLs from LongURLs), as well as an [[API Gateway]] which handles routing (and any  other cross-cutting concerns)
![[Pasted image 20250523131956.png]]
- And we can scale our Read Service horizontally, and far fewer of these Write Services!

There's a very little bit of code happening on these two boxes, and splitting this up into two means you have two services to maintain, and it's probably overkill. You could very well keep them in the same box and just scale them the same. This is a tradeoff to weigh that you can discuss with your interviewer.

On modern cloud services, you have autoscaling foreach of these services:
- We configure Memory and CPU limits, saying "If 75% of my memory is used, throw up a new box! If <20% is used, take a box down! Adn these will scale automatically"

Technically yes there is a [[Load Balancing|Load Balancer]] in front of them, routing with (e.g.) [[Round Robin]], etc... but in practice these days, this is just autoscaling...

But this Horizontal scaling ==**Incurs some important implications for our Counter, which we had in our Write Service!**==
- We need all of our write services to agree on the same current count!
![[Pasted image 20250523132338.png]]
==We need to pull that counter off of the instances -- that counter can't be in memory any more on these servers!=
- We need some sort of **global counter**...
	- This could be Redis. 
	- [[Redis]] has an `INCR`, and because ==Redis is single-threaded, we don't have to worry about any concurrency problems==
	- This means how that in order to get the count, you have to go somehwer else, make a network call, etc.
	- Something we could do that's kinda cute is:
		- ==When a server in WriteService comes online, it can grab the next 1,000 counts from the Redis counter==
![[Pasted image 20250523133101.png]]
Great!

Now let's talk about the Database, how does this need to scale?
- How much does this take..
	- Short URL: 8 bytes
	- LongURL: 100 bytes
	- Creation Time: always 8 bytes
	- Custom alias: Up to 100 bytes
	- Expiration time: 8 bytes
This is... about 200 bytes, let's maybe round up if we want to throw more stuff in there, to 500 Bytes times 1B users, to 500GB
- 500GB isn't a lot; **SSDs are in the hundreds of TBs!**

What bout the Read Throughput?
- We put most of the reads on Redis, in our Cache. So our DB is probably chillin!
- If we needed to scale the database, we might [[Partition]], perhaps using the ShortURL as the partition key.
	- This would just mean we would have multiple instances of our database, and we'd take our shortURL, modulo 3, and then store it on our databases.
- But in this case, we don't need to! We did the math and it doesn't matter.



### DD for NFR: High Availability
- For Redis... if Redis goes down, it's fine! It's Read Through... so we'll load it back up by just hitting the database. We'll be slow for a bit, no big deal.
- If the Global Counter goes down, we'd want to have some HA modes here, meaning we'd have some redundancy.
	- WE could periodically snapshot the count and put it to disk
	- ==**Lookup Redis High Availability Mode**==
- In the Database, we could have some replicas
	- We could just have a single replica in case the DB went down... but we can also take snapshots every hour or so, stored in something like [[Amazon S3]], so that if the DB goes down, we can just pull it back up.
	- With the replicas there, if the DB goes down, we can point to the Replica until the main one gets back up using the snapshot.


