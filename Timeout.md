
When clients are connecting to a server, we want to make sure that they eventually time out.
The worst possible scenario: 
- We get some acknowledgement from a server that our work is in progress, and then never hear back.
- We want to put a timeout that isn't too short and isn't too long (long enough that servers can respond, but not so long that it doesn't serve its purpsoe).

When there's a failure detected by timeout, we're going to [[Retry]] our request. We try again.

A naive approach would be: Every 5 seconds we go and make a retry. This is naive, because it suffers from this bunching behavior; if I have a service that can only handle 2 transactions per second, and we have 3 clients making requests at a time... they all fail, and they have a policy to retry again in 5 seconds, and they keep trying/failing together.

We can employ [[Backoff]] (often times [[Backoff|Exponential Backoff]]), which means that the time between successive retries increases exponentially. This has a two-sided purpose: Our first retry should happen pretty quickly.  We don't want our users to wait too long for something that's transient, but if we hit (e.g.) 2 retries in a row, then something is probably wrong, and it'll probably take some time to recover.

This is nice -- we're giving the server a chance to recover, but we aren't handling the synchronization problem! So we need to add some [[Jitter]], where we add some random amount of time. This ensures the synchronization doesn't happen, and that successive retries are also spread out.

Might look like:
```
delay_n = random(0, min(cap, base * 2^n))
```

In interviews, the gold standard is: "Timeouts with Retries with Exponential Backoff and Jitter"



![[Pasted image 20260605175027.png]]


See: [[Retry|Retries]], [[Backoff]], [[Jitter]]