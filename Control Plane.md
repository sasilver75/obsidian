Compare with: [[Data Plane]]


The part of a system that ==manages configuration, policy, and state.==
- Handles things like provisioning resources, distributing configuration, enforce access policy, electing leaders, and reacting to changes in desired state.
- Work here is typically low-volume but high stakes; correctness, consistency, and durability typically matter more than latency.
- It's often centralized (or [[Quorum]]-based), and backed by a [[Strong Consistency]] core.


Rule of Thumb: If it runs on every request, it's [[Data Plane]]. If it runs when something *changes*, it's [[Control Plane]]

