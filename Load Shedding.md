
When a system intentionally rejects or drops some work during overload so that it can keep serving the most important requests. 

Instead of accepting every request and becoming slow or crashing, the system says "no" to lower-priority traffic.

Examples:
- Returning `503 Service Unavailable` when too many requests arrive
- Drop background jobs before user-facing requests
- Rejects expensive search queries during peak traffic
- Stop accepting new video streams while preserving existing ones

The goal is [[Graceful Degradation]]: We want to protect the system's core functions by sacrificing less important work.


# Relationship with [[Backpressure]]
- Both are overload-control techniques, but they respond differently.
- [[Backpressure]] slows the producer down
	- A service says "I'm busy, please send requests more slowly!" 
		- This might happen through queues filling up, rate-limit responses, TCP flow control, stream buffering, or a message broker delaying consumers/producers.
	- Producer is producing too much work, so the consumers say: "Slow down!"
- [[Load Shedding]] drops or rejects work.
	- A service says: "I'm busy, I'm not accepting this request." It might return a `503`, discard low-priority jobs, or reject new connections.
	- Producer is producing too much work, so the consumers say: "No."
- ==They often work together==
	- First, use backpressure to reduce incoming rate
	- If the system is still overloaded, use load shedding to protect yourself.
