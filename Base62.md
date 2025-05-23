
Uses 0-9, A-Z, a-z

The use case that I've seen for this is in the context of URL shortening, where we want to turn a URL like https://www.canoeclub.com/sale into a ShortURL
- To keep it short (like 5-7 characters, so someone can even type it in if they wanted), we want to somehow encode it into a more dense format.
- The reason that we don't use the 
- So what we can do is is **something like**:
	- Have a Counter that we increment for each URL that we make.
	- Take the Count and Base62 encode it. 
		- This turns 1 into n
		- This turns 2 into o
		- This turns 3 into p
		- This turns 53 000 152 242 into 5BWk3F8WT706atW
- The problem is that for this strategy above, it's pretty transparent what the "next" URL is going to be. So if an adversary can count and use Base62 encodings, they can "scrape" all of your URLs (assuming you aren't rate limiting, etc.)
	- So we can use special libraries like **Sqids** or **HashIds** that use obfuscatory functions (that are still [[Bijective]], like Base62 encoding), but that make the sequence look random to and hard to enumerate in the same way by attackers.
