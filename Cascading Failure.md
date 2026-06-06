


If you've ever been up at a 3AM on call with a server that just won't come up for a variety of reasons, you probably are sensitive to Cascading Failures and the effect they have on your sleep and your life.

Sometimes problems in one part of the system can create problems in another part of the system!


![[Pasted image 20260605180216.png]]
- Server A makes a request to Server B, which queries the database.
- One day, our admin guy decides: "We need to backup the database!" He triggers a database snapshot.
- Immediately, our database is operating at ~50% efficiency.
- The Server B developer wanted to be robust to failures in the network, so they put [[Timeout]]s on their requests to the database. Now that the database is returning 50% slower, it's now failing those requests.
- Like a good service, it's retrying those requests ... but those retries are also failing. It tries it 3 times, progressively longer ([[Backoff|Exponential Backoff]], [[Jitter]]), but ultimately fails.
- Server A is making request to Server B; for those requests that fail, it's doing it again.
- So... Because of the snapshot, our Database is degraded, and our first server starts to fail, and the second server punishes it for the underperformance of the database.
- If we think the problem is server B, we might start it up again, and it will go back into the same failure mode.
- What we need is for server A to give server B a break, so that server B gives the database a break.

The way to deal with this is to implement a [[Circuit Breaker]]
- In your house, when the electricity in your house exceed a certain current, it flips off, and requires human intervention. 
- In software, a circuit breaker trips when failure exceeds a certain level, and periodically resets itself.
- If we put one between Server A and Server B... (it's really inside Server A)...
- Server A will say: "Server B is obviously not handling many of my requests. I'm going to temporarily pause, failing my requests... and give it a chance to recover before trying again."
- From an operational perspective, this gives me a chance to notice that Server B is slow, and it's not answering its request quickly because of the database, and I can go and fix the database problem.
- Later, the circuit breaker resets, and things go on as usual.

Circuit Breakers build robust failures by failing in an automated way (and you'll often want to have Alarms/Alerts on these starting!)

