---
aliases:
  - Consensus
---
The problem of making several machines agree on one value, even when some machines crash, messages are delayed, messages are reordered, and different machines temporarily see different parts of the system.

Consensus is any protocol ([[Raft]], [[Paxos]], etc.) that lets a group of replicas choose one value such that once any value is chosen, no different value can ever be chosen. It's a way of building a [[Log]] over many replicas. This value might be:
- "Node A is the leader for term 12"
- "The next log entry is `set x = 5`"
- "Transaction T is committed."
- "Configuration C is the active cluster membership."

In practice, systems usually run consensus ***repeatedly*** to build a replicated log. 

Example Problem: Suppose five database replicas receive conflicting requests:
- Client 1 asks to write `x=5`
- Client 2 asks to write `x=9`
- The network delays some messages
- One replica crashes
- Two replicas temporarily think that different leaders exist

Consensus prevents the system from ending up with one subset of replicas believing `x=5` was committed, while another subset believes `x=9` was committed for the same logical slot.

> If the system decides, it decides only one value. Think of Consensus as a very cautious election for a value.

A value is not decided just because one server likes it; a value is typically decided because a sufficiently large overlapping group of servers has recorded enough evidence that this value won. This overlapping group is usually a [[Quorum]] (in a 5 node cluster, a common quorum is 3 nodes). The key property is that any two majorities intersect in at least one node.

Typical vocabulary:
- ==Replica==: A server participating in consensus
- ==Proposal==: A candidate value that might be chosen
- ==Round/term/epoch/ballot==: A monotonically-increasing attempt number used to order competing attempts.
	- ((I think this is also a [[Fencing Token]]?))
- ==Quorum==: A subset large enough to make decisions, usually a majority
- ==Vote/accept==: Durable evidence that a replica supports a value in a particular round
- ==Chosen/committed==: A value has enough votes that no conflicting value can later be chosen
- ==Leader==: A replica that coordinates proposals for efficiency. Not conceptually required, but practically common.

A simplified single-value consensus protocol typically looks something like:
1. A coordinator starts a numbered round: "I want to run round 17, has anyone already accepted a value in an earlier round?"
2. A quorum responds, with either "I have not accepted anything," or "I previously accepted value V in round N." The replica may also promise not to accept proposals from older rounds.
3. The coordinator preserves any values that might already have been chosen. If no prior value appears, a coordinator may propose its own value, but if a prior accepted value appears, the coordinator must usually propose the highest-round previously accepted value it heard about. This means that a new leader can't overwrite a value that may already have been chosen by an earlier leader.
4. The coordinator asks a quorum to accept the selected value, saying "Accept value V for round 17." 
5. If a quorum accepts, the value is chosen. Once a quorum has accepted V for that round, V is decided. Other replicas may learn V later, but the decision has already become irreversible.


Q: Why is quorum intersection enough?
A: Assume five replicas: A B C D E
- A value needs three votes.
- Suppose `x=5` was chosen by A B C
- A later conflicting value `x=9` would also need three votes; any group of three must overlap with A B C.
- At least one replica from the old deciding quorum participates the new quorum.
	- This overlapping replica carries information or promises that force the later round to respect the earlier possible decision.
	- This is why consensus protocols care so much about durable votes, terms, and "what was the highest accepted value you've seen?"

The hard part is preserving safety during ugly timing cases:
- A leader send accept messages to three replicas, but crashes before telling anyone the value was chosen.
- A new leader appears and doesn't know whether the previous value was chosen.
- Some replicas accepted a value; others have never heard of it.
- Messages from an old leader arrive late, after a new leader is active.
- Two leaders believe they are legitimate because of a partition or timeout race.

Consensus solves these cases by making every new round recover the evidence left by older rounds before it is allowed to choose a value!

# Safety Versus Liveness
- Consensus has two major families of properties:
	- ==Safety==: Nothing bad happens; not two different values are chosen for the same decision.
	- ==Liveness==: Something good eventually happens; a value is eventually chosen.
- Safety is usually unconditional under the protocol's failure model. Even with arbitrary delays/retries/partitions, the protocols should not choose two conflicting values.
- Liveness is conditional; a consensus system usually needs something like:
	- A majority of replicas are alive
	- Messages between majority replicas are eventually delivered
	- Clocks or timeouts eventually become useful enough to stop constant leader churn
	- The storage used for votes does not lose promised or accepted state



This distinction matters because of the [[FLP Impossibility Theorem]]: in a fully asynchronous distributed system with even one crash failure, deterministic consensus cannot ***guarantee*** termination in all executions. Real protocols work by adding practical assumptions, usually around eventual synchrony, timeouts, stable leaders, randomized backoff, or external failure detectors.


The central invariant is:
> Once a value might have been chosen, every future successful decision path must preserve that value.

Different protocols encode this invariant differently:
- [[Paxos]] uses ballots, promises, accepted values, and quorum intersection.
- [[Raft]] uses terms, logs, leader completeness, and election restrictions.
- [[Zookeeper Atomic Broadcast|ZAB]] uses epochs, primary order, and recovery synchronization.
- [[KRaft]] adapts Raft-style metadata quorum behavior for Kafka.
All protect the same thing: Future leaders must not be allowed to erase decisions that could already have become committed.

Bottom Line: Consensus works by combining...
1. Monotonic rounds or terms to order competing attempts.
2. Quorums whose intersections preserve information across attempts.
3. Durable votes so crashed replicas remember what they promised or accepted.
4. Recovery rules that force new leaders to carry forward possibly chosen values.
5. Commit rules that declare a value chosen only after enough replicas have recorded it.














_______________

Sometimes it's unacceptable to lose state, even in the face of hardware failures
- Who's the leader? Who's holding the lock?
The way that we ensure this data remains around in the face of hardware failure is via using distributed consensus algorithms.
These algorithms are slow, and we only use them for important application state.


Consensus algorithms allow us to build a log over many replicas.