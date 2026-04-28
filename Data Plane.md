
Compare with: [[Control Plane]]

The part of a system that ==handles actual workload traffic==.
- It executes the operations users and services actually care about, on every request.
- Throughput, latency, and availability dominate the design.
- It runs continuously, scales horizontally with load, and is usually distributed close to where work happens (per-node, per-region, per-pod).

This might be a [[Kubernetes]] kubelet starting containers, or an [[Amazon EC2|AWS EC2]] VM serving traffic, an S3 GetObject, a DynamoDB read.

Rule of Thumb: If it runs on every request, it's [[Data Plane]]. If it runs when something *changes*, it's [[Control Plane]]