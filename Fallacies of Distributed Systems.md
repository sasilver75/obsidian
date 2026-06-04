---
aliases:
  - Fallacies of Distributed Computing
---

Originally listed fallacies:
1. The network is reliable 
2. Latency is zero
3. Bandwidth is infinite
4. The network is secure
5. Topology doesn't change
6. There is one administrator
7. Transport cost is zero


The effects of the fallacies:
- Applications are often written with little error-handling on networking errors. During network outages, such applications may stall or wait indefinitely, permanently consuming memory or other resources.
- Ignorance of network latency induces application-layer developers to allow unbounded traffic, greatly increasing dropped packets and wasting bandwidth.
- Ignorance of bandwidth limits on the part of traffic senders can result in bottlenecks.
- Complacency regarding networking security results in being blindsided by malicious users and programs that continually adapt to security measures.
- Changes in network topology can have effects on both bandwidth and latency issues, and therefore can have similar problems.
- If a system assumes a homogenous network, it can lead to the same problems that result from the first three fallacies.
















