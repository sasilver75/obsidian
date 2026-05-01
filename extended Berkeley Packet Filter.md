---
aliases:
  - eBPF
---
A [[Linux]] kernel technology that lets you run sandboxed programs inside that [[Kernel]], without writing kernel modules or recompiling the kernel.

Core Idea:
- Write a small program (typically in a restricted C dialect)
- Compile it to eBPF bytecode
- The kernel *verifies that it's safe*, then JIT-compiles it to native code and attaches it to a *hook point* in the kernel.
- When that hook fires (a syscall, a packet arriving, a function entry), your program runs.


# Disrupting the [[Sidecar]] pattern
- In a traditional [[Service Mesh]] (e.g. [[Istio]]), every pod gets a sidecar proxy ([[Envoy]] typically) injected next to the app container.
- All traffic in/out of the pod is transparently redirected through that proxy via [[iptables]] rules. 
	- The proxy handles [[Mutual TLS|mTLS]], [[Retry|Retries]], [[Load Balancing]], Observability, Traffic policy
- The cost:
	- One proxy per pod (1000 pods = 1000 Envoys, each eating RAM and CPU)
	- Two extra hops per request (in and out additional hop), adding latency
	- [[iptables]] redirect is clunky: Every packet bounces between [[Kernel Space]] and [[User Space]]
	- Lifecycle pain: Side car has to start before the app, shut down after, and version-skew with the control plane

==With eBPF you can do most of what the sidecar did in the kernel, attached to the pod's network namespace, with no extra process.==
- The slogan is =="service mesh without sidecars"==
- This is what [[Istio]]'s **ambient mode** (released GA 2024) is pursuing/does. The sidecar-per-pod model is on its way out for most workloads.