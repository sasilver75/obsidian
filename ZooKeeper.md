
A tool often used as a [[Coordination Service]] in distributed systems, used to help:
- Elect leaders
- Maintain consistent configurations
- Detect failures in real time
- etc.

Released in 2008, so it's somewhat aged and numerous alternatives have emerged, but it remains central to the Apache ecosystem in particular.
- Still, **understanding ZooKeeper teaches essential distributed systems concepts that apply even if you don't use it directly!**
- By learning how ZooKeeper handles coordination through simple primitives (detailed below), you gain insights into solving universal problems like [[Distributed Consensus|Consensus]], [[Leader Election]], and Configuration Management. These primitives include:
	- Hierarchical Namespaces
	- Data Nodes
	- Watches







# Motivating Example
- Imagine that we're building a **Chat Application**
- Initially our Chat App runs in a single server, and life is simple; When Alice sends a message to Bob, both users are connected to the same server, and the server knows exactly where the deliver the message; it's all in=memory, low latency, no coordination needed.

![[Pasted image 20250524190247.png]]
















