
The Circuit Breaker pattern is a resilience pattern used to prevent an application from repeatedly calling a failing dependency.

It works like an electrical circuit breaker:
- When failures become too frequent, the circuit "opens" and stops sending requests to the failing service for a while, preventing wasted work, cascading failures, thread exhaustion, long timeouts, and degraded user experience.

Three States:
- ==Closed==: Requests flow normally. The breaker tracks failures, timeouts, or error rates.
- ==Open==: Too many failures have occurred. Requests now fail fast instead of actually calling the dependency.
	- You return some sort of cached/default response, or just "Payment service is temporarily unavailable."
- ==Half-Open==: After a cooldown period, the breaker allows a *small number* of test requests. If these requests succeed, it closes again, and requests flow normally. If these requests fail, the circuit breaker opens again.

Circuit breakers are commonly used around:
- HTTP calls to external services
- Database calls
- Message brokers
- Third-party APIs
- Slow or unreliable internal services

These are usually paired with [[Timeout]]s, [[Retry|Retries]], [[Bulkhead]]s, and fallbacks. The important distinction is that retries keep trying; a circuit breaker decides when to stop trying for a while.





