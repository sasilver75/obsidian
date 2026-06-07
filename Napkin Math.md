
A reference for useful napkin math in system design interviews.
These are general order-of-magnitude references, not exact constants.


# Rules of thumb:
- [[RAM]] is ~1,000x faster than [[Solid State Disk|SSD]] random read.
- [[Solid State Disk|SSD]] random read is ~100x faster than an [[HDD]] random read.
- Same-region service calls are ms-scale
- Cross-country calls are 10s of ms
- Cross-continent calls are 100s or ms
- Disk, network, and external APIs dominate latency far more than CPU time.

### CPU/Memory
- [[L1 Cache]] access: ~1 ns
- L2 cache access: ~3-5 ns
- L3 cache access: ~10-20 ns
- Main memory/RAM access: ~50-150 ns
### Storage
- NVMe [[Solid State Disk|SSD]] random read: ~50-200 us
- [[HDD]] seek/random read: ~5-10 ms
- SSD sequential read: ~500MB/s - 7 GB/s
- HDD sequential read: ~100-250 MB/s
### Network Latency
- Same machine: <1-100 us
- Same rack: 0.1-0.5 ms
- Same datacenter: 0.5-2 ms
- NYC to LA: 60-80 ms RTT
- US East to London: 70-100 ms RTT
- US to India: 180-250 ms RTT
- Worst normal global RTT: 250-350 ms
### Connection Setup (1 RTT = client -> server -> client)
- [[Transport Control Protocol|TCP]] handshake: 1 RTT
- [[Transport Layer Security|TLS]] 1.3 handshake: 1 RTT after TCP
- [[Domain Name Service|DNS]] recursive lookup: ~20-100 ms
### Caches / Databases
- In-process cache read: ns-us
- Redis/Memcached same AZ: ~0.5-2 ms
- Redis/Memcached cross-region: 10s-100+ ms
- Simple DB query: ~5-20 ms
- Complex DB query: ~50-500+ ms
