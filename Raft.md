A [[Distributed Consensus]] algorithms designed for understandability w.r.t. more complicated options like [[Paxos]]. Raft is a crash-fault-tolerant consensus protocol where a cluster elects a leader, the leader replicates an ordered [[Log]] of commands to a majority of servers, and committed log entries are guaranteed to sruvive future leader changes.

It's built around a simple operating story:
> Elect one leader, make the leader choose the log order, and make every future leader prove it has the committed history before it can lead.

That's the whole shape. Instead of presenting consensus as many independent ballots like Paxos often does, Raft organizes the protocol around ==terms==, ==leaders==, and a ==replicated log.==

Raft is often used for state machine replication:
```
log index 1: create user 123
log index 2: set user 123 plan = "pro"
log index 3: revoke token abc
```
If every server applies the same log entries in the same order, every server reaches the same state. Raft's job is to make the cluster agree on that log despite crashes, delayed messages, and leader changes.

Mental Model: Raft is like a parliament with a clerk.
- The leader is the clerk who decides the next page of the official record.
- The followers copy the clerk's record.
- A record entry becomes official only after a majority has copied it.
- If the clerk disappears, the group elects a new clerk.
- A candidate can become clerk only if the candidate's record is *at least as up to date* as the voter's records.
	- This is the safety trick: Raft doesn't merely elect any reachable server, it elects a server that cannot have missed committed history.


Components:
- Server: A replica participating in the cluster
- Follower: Passive server that responds to leaders and candidates.
- Candidate: Server trying to become leader.
- Leader: Server currently coordinating log replication.
- Term: Monotonically-increasing logical epoch.
- Log entry: A command plus the term in which the leader created it.
- Commit index: Highest log index known to be committed.
- Majority quorum: More than half the servers, e.g. 3 of 5.

Each server stores durable state such as:
- `currentTerm`: The highest term that the server has seen.
- `votedFor`: Which candidate the server voted for in the current term, if any.
- `log`: The ordered sequence of log entries.


# Process
1. Leader Election
	- Each server starts as a follower and expects periodic heartbeats from a leader.
	- If a follower stops hearing from a leader, it starts an election by incrementing its term, voting for myself, and asking other servers for votes.
	- A candidate becomes leader only if it receives votes from a majority, and voters prefer candidates whose logs are at least as up to date as their own.
2. Terms
	- A term is Raft's monotonically-increasing logical epoch number.
	- Terms let servers recognize stale leaders and stale messages: if a server sees a higher term, it updates its own term and steps down to follower.
	- At most one leader can be elected per term, because each server can vote for only one candidate in that term.
3. Log Replication
	- Clients send commands to the leader, and the leader appends each command as a log entry.
	- The leader then sends `AppendEntries` messages to followers, including the previous log index and previous log term so followers can verify that their log matches the leader's prefix.
	- If a follower has conflicting entries, the leader backs up to the last matching point and overwrites the follower's uncommitted suffix.
4. Commitment
	- A log entry becomes committed when the leader knows that the entry is stored on a majority of servers.
	- Once committed, the entry is safe to apply to the replicated state machine, and the leader tells followers the latest committed index in later messages.
	- Raft protects committed entries across leader changes by ensuring the future leaders must have sufficiently up-to-date logs to win elections.




