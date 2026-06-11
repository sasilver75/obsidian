
When different machines disagree about the current time.

```
Server A thinks it is 10:00:00
Server B think its it 10:00:07
```

These clocks are skewed by 7 seconds!

This matters because many things in distributed systems depend on time:
- Expiring [[Session]]s or [[Access Token]]s
- Ordering events
- Database writes and timestamps
- Cache expiration
- Logs and debugging
- Leader election or distributed locks


Common causes:
- Hardware clock drift: Each machine's physical clock (e.g. quart oscillator) runs slightly fast or slow.
- Bad or missing time synchronization (Maybe [[Network Time Protocol|NTP]] is disabled/misconfigured/blocked by firewall)
- ...

A little skew is normal. Distributed systems should be designed to tolerate it instead of assuming that every machine has exactly the same time.

