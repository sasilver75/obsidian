---
aliases:
  - Consensus
---
e.g. [[PAXOS]], [[RAFT]]



Sometimes it's unacceptable to lose state, even in the face of hardware failures
- Who's the leader? Who's holding the lock?
The way that we ensure this data remains around in the face of hardware failure is via using distributed consensus algorithms.
These algorithms are slow, and we only use them for important application state.


Consensus algorithms allow us to build a log over many replicas.