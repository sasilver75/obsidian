---
aliases:
  - RDMA
---
A high-performance networking technology that allows computers to transfer data directly from one machine's memory to anothers', without involving the CPU, operating system, or cache of either machine.

This bypasses the [[TCP/IP]] stack, significantly reducing latency and CPU overhead while increasing bandwidth between machines for data-intensive tasks.
- Traditional networking requires the CPU to manage data packets, which introduces delays; RDMA offloads this transfer entirely to the network adapter (NIC), freeing the CPU for application processing.

RDMA is implemented over different fabrics:
- InfiniBand: Native, low-latency fabric
- RoCE (RMDA over Converged Ethernet): Enables RDMA over Ethernet networks
- iWARP: Enables RDMA over TCP
- RDMA over Thunderbolt introduced in macOS 26.2, effectively allowing multiple Apple silicon maps to act as a single clustered node. Optimized for Thunderbolt 5.

Commonly used in high-performance computing clusters (HPC), data centers, and AI systems.


