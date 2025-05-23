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

Let's look at our APIs:
- POST /urls
	- Client is going to make an API request of getShortURL() to hit our Primary Server, which will have some sort of 









