---
aliases:
  - 2PC
---


A strategy used for [[Distributed Transaction]]s.
- Ensures that a transaction either **commits** on all participating nodes, or **rolls back** entirely, guaranteeing [[Atomicity]] and Consistency (in the [[ACID]] sense). 
- Commonly used in distributed database system and microservice architectures to handle distributed transactions.

![[Pasted image 20250523220300.png]]


- Protocol consist of two phases:
	- The ==commit-request phase==, also sometimes called the (==prepare phase== or ==voting phase==), in whicha coordinator process attempts to prepare the transactions coordinating participants to take necessary steps. for either committing or aborting the transaction, and to voite either "Yes" (commit) or "No" (abort).
		- Coordinator sends a query to commit message ot all participants, waits until it hears a reply from all participants
		- Participants execute the transaction *up to the point where they would commit*, with each writing an entry to their **undo log** and their **redo log**.
		- Each participant replies with an agreement message if the actions succeeeded, or an abort if they experienced a failure that makes it impossible to commit.
	- The ==commit phase==, which, based on reporting of participants, the coordinator decides whether to commit (only if all vote Yes) or abort the transaction (otherwise), and notifies the result to all participants, who then follow with the requisite actions (commit or abort) with their local transactional resources.
		- If all "Yes" are received:
			- Coordinator sends a commit message to all participants
			- Each participant completes the operation, releasing all locks and resources held during transaction
			- Each participant sends an ack to the coordinator
			- The coordinator completes the transaction when all acks are receieved
		- If any "No" is received:
			- Coordinator send a "rollback" message to all participants
			- Each participant undoes the transaction using the **undo log**, and releases the resources adn locks held during the transaction
			- The coordinator undoes the transaction when all acks have been received

The protocol assumes:
1. There is stable storage at each node with a [[Write-Ahead Log]]
2. No node crashes forever
3. The data in the WAL is never lost or corrupted in a crash
4. Any two nodes can communicate with eachother.


**Disadvantages**
- The greatest disadvantage is that it's a blocking protocol!
- If the coordinator fails permanently, some participants will never resolve their transactions! 
- A 2PC commit protocol can't dependably recover from failure of both the coordinator and a cohort member during the commit phase...