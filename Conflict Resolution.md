
==The process of deciding what to do when two or more writes produce competing versions of the same logical data.== 

Answers:
- Which value wins?
- Can both changes be merged?
- Should one write be rejected?
- Should a human decide?
- Should we create a compensating action?

Needed in distributed database systems when two or more writes can happen without first agreeing on one global order, creating a [[Write Conflict]]. 

[[Strong Consistency]] tries to prevent conflicts *before commit.*
[[Eventual Consistency]] often accepts writes first, and then resolves conflicts afterwards.



# Common Conflict Resolution Strategies

There are two broad appraoches:
- ==Conflict prevention==: Coordinate before accepting writes.
	- A bank transfer should use conflict prevention: [[Serializable Isolation|Serializable]] [[Transaction]]s, [[Strong Consistency|Linearizability]] for balances, a ledger with strict ordering.
	- Give strong correctness, simpler reads, fewer surprises at the cost of more latency, less availability, harder multi-region scaling.
- ==Conflict resolution==: Accept writes independently, then merge/reject/repair later.
	- Automatic resolution: fast, good UX when rules are correct, but risk of lost updates/subtle bugs
	- Expose to users: Avoids silent data loss, but slower UX and annoying if frequent


Examples include:
- **==Avoid conflicts, using a single writer==**: Route all writes for a single item/key/partition through one leader, who gives writes a single order.
	- Bank balance, inventory, unique IDs, locks. Give strong correctness, but you're less available during partitions and often higher latency.
- ==[[Optimistic Concurrency Control]]== (OCC): Every record has a version. Client says "update this only if version is still 7." If someone already changed it to version 8, reject or retry.
	- Forms, settings, admin edits, APIs. Prevent silent overwrites, but users may get annoyed by "'someone else changed this" errors if there's high contention for things they're trying to update.
- ==[[Last Write Wins]]:== Pick the write with the latest timestamp or highest version, and discard the others.
	- Caches, presence status, simple replaceable values. Simple, but can lose real updates, dangerous for important data. Affected by [[Clock Skew]].
- [[First Write Wins]]: Accept the first write and reject later conflicting writes.
	- Seat reservation, username claims, idempotency keys. Good for claims, but requires a reliable ordering authority.  ((I DISAGREE that this would be good for seat reservation; imagine getting a confirmation that you got a seat, and then later you get overwritten by an "earlier" claim of that seat!))
- Field-level merge: Merge non-overlapping changes to different fields (conflicts still can exist)
	- Profiles, document with independent fields. Conflicts can still exist, so this isn't IMO a complete strategy.
- ==Application-specific merge==: App defines domain rules (e.g. shopping cart additions are unioned, tags are merged)
	- Carts, tags, preferences, collaborative apps. Often the best UX, but requires careful domain-specific logic.
- ==Commutative operations==: Store operations that can be applied in any order and still converge, e.g. "increment by 1" instead of "set count to 42"
	- Counters, likes, metrics, some ledgers. Works only for operations with safe merge semantics.
- [[Conflict-Free Replicated Datatype|CRDT]]s: Data structure designed so replicas can merge automatically and deterministically.
	- Offline-first apps, collaborative editing, distribute counters, sets. Excellent convergence but complexity can be high.
- [[Operational Transform]] (OT): Transform concurrent edits so that they can be appleid consistently.
	- Google Docs-style editing. Powerful but complex and usually domain-specific.
- ==Manual/User-assisted resolution==: Keep both versions and ask a user to choose or merge.
	- Legal docs, design files, high-value edits. Preserves correctness, but interrupts users and does not scale for frequent conflicts.
- ==Compensation/reconciliation workflows==:  Allow a temporary conflict, then repair with a followup action, like if you oversell inventory, then cancel/refund one order.
	- Booking, marketplaces, payment-adjacent workflows. High availability, but the actual business process must tolerate correction.



# How do some of these actually play out in reality? The detection and resolution
- Two cases:
	- Conflicts caught before a write commits
	- Conflicts discovered later, after multiple versions already exist

A database usually does *mechanical conflict detection*, while the application does *semantic conflict resolution.*
- The DB can tell: "These two writes were based on the same old version" and "these two versions were created concurrently," but it cannot know "For a shopping cart, adding item A and item B should both survive."  That latter domain rule should live in application logic.

#### [[Optimistic Concurrency Control|OCC]] example (Conflict Prevention, application-level resolution):
Cart is stored as:
```
cart_id = 123 
version = 7
items = [coffee]
```
Two clients both read `version 7`, then:
client A submits
```
base_version=7
add tea
```
client B submits
```
base_verison = 7
add sugar
```
The app tries a conditional update:
```
  UPDATE carts
  SET items = ..., version = version + 1
  WHERE cart_id = 123
    AND version = 7;
```
A succeeds and creates `version 8`.
B's update affects `0 rows`, because the cart is no longer version `7`.
Now the application knows: "B wrote based on stale data!" (after perhaps checking that indeed that cart *does* exist)
Instead of blindly failing the update, the app can do an application-specific merge:
```
latest cart: [coffee, tea] 
B intended: add sugar 
merged cart: [coffee, tea, sugar]
```
Then it writes `version 9` with that update.
So in this case the database didn't understand carts, it just enforced the version check.



#### Multi-Replica example (Conflict Resolution)
- In an eventually-consistent multi-region system, perhaps two writes commit independently:
```
US replica accepts: add tea
EU replica accepts: add sugar
```
Later, replication discovers that neither write happened "after" the other. They're concurrent.
The database may detect this using metadata like:
```
US version: {us: 8, eu: 3}
EU version: {us: 7, eu: 4}
```
Neither dominates the other, so there's a conflict.
Then, one of three things usually happens:
1. On write: Application detects stale version during a write request and immediately runs merge/resolution logic before committing or rejecting the write.
2. On read: Database returns multiple conflicting versions; app merges before showing the user. This usually means the conflict is discovered lazily when someone reads the record (after the conflict is detected asynchronously in the background), when something like the following is returned, and then the application merges them using (e.g.) domain rules.
	```
	{
	    "conflict": true,
	    "versions": [
	      { "items": ["coffee", "tea"], "context": {"us": 2, "eu": 1} },
	      { "items": ["coffee", "sugar"], "context": {"us": 1, "eu": 2} }
	    ]
	}
	```
3. Background resolver: A worker consumes conflict records, merges them, writes the resolved value

Commonly: 
1. The database stores version/revision metadata
2. The app detects stale writes or concurrent revisions
3. The app applies a merge function
4. The app writes back the resolved value





# Simple Example
Example: Two users editing the same profile at the same time. (multi-writer scenario)
Initial value:
```
display_name = "Alice"
version = 1
```
Two replicas or clients read version 1, then:
Client A writes:
```
display_name="Alice Smith"
```
while Client B writes:
```
display_name="Alice Johnson"
```
If both writes are accepted independently, the system later has two competing versions!

Possible resolutions:
- [[Last Write Wins]]: Whichever write has the later timestamp wins, the other is lost.
- [[Optimistic Concurrency Control]] (OCC): One write succeeds, the other fails because version changed.
- Manual Mege: User is asked to choose between "Alice Smith" and "Alice Johnson"
- App-specific Rule: Maybe prefer verified legal name over nickname, depending on domain ((bad example)).

