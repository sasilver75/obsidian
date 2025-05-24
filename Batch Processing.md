See also: [[Stream Processing]]

These are performed "offline," often triggered by some timer, manually, or in response to to the completion of a predecessor batch job.

Batch Processing Frameworks like [[Spark]] abstract many difficulties (hardware failures, load balancing, concurrency bugs) for us.
- Specify operators to modify the data
- Specify how you want data to be partitioned
- Allow checkpointing intermediate state to avoid restarting whole jobs if one node fails
- Try to perform computations where data already is stored to avoid unnecessary network usage (take advantage of data locality)

