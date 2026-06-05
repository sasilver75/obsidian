
If event B depends on event A, everyone who sees B must *also* see A *first*. In other words, causally related events appear in the same order for all users, ensuring logical ordering of dependent actions (e.g. post comment on a post appear after the post).
- *Concurrent events* can appear in different orders.

# How is it achieved?
- The system has to track dependencies between operations.
- If write B was created after reading write A, then B carries metadata saying "B depends on A."
- Replicas delay making B visible until they have also received and applied A.
- This is often implemented using [[Vector Clock]]s, [[Version Vector]]s, Causal broadcast, or [[Conflict-Free Replicated Datatype|CRDT]].

# Tradeoffs
- Preserves cause-and-effect without total global ordering.
- Less coordination than linearizability, but more metadata and implementation complexity.
- Concurrent conflicts still need [[Conflict Resolution]].

# Use Cases
- Chat, comments, threaded conversations, collaborative activity feeds
- Cause and effect must be preserved. If someone *replies* to a message, anyone who sees this reply should *also* see the original message! But two unrelated messages can appear in different orders for different users.

