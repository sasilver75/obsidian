


Tradeoff when choosing a [[Message Queue]] partition key:
- Often times ordering of messages (which might be a requirement) is only guaranteed within a partition. So you select your partition key (e.g. `user_id`) such that messages that need to be ordered are in the same partition. At the same time, partition keys are also used to distribute load across partitions. It might be the case that `user_id` does not distribute evenly, and hot keys can make this more pronounced. The key that you need 
	- Straightforward design rule: ==Choose the narrowest partition key that preserve the ordering guarantee that the system actually needs.==

