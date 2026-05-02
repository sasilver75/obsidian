---
aliases:
  - IDS
  - Network IDS
  - NIDS
  - Host IDS
  - HIDS
---
A security tool that monitors traffic or system activity for signs of attack, and raises alerts if it sees something suspicious. The key word is *detection*: IDS watches and warms, it doesn't block.

Its sibling, [[Intrusion Prevention System]] (IPS) does the same monitoring, but sits inline and blocks malicious traffic in real time.
- You commonly hear [[Intrusion Detection System|IDS]]/[[Intrusion Prevention System|IPS]] together, most modern products do both, configurable per rule (alert-only vs block).

IDS has two main flavors:
- Network IDS (NIDS)
	- Watches traffic flowing across the network, usually via a SPAN/mirror port on a switch, a network tap, inline deployment. Examples are Snort, Suricata, Zeek, AWS GuardDuty
- Host IDS (HIDS)
	- Runs on individual servers and watches what's happening *on* the box: file integrity, log entries, process behavior, system calls, registry changes (Windows)

Modern systems blend [[Intrusion Detection System|IDS]]/[[Intrusion Prevention System|IPS]], [[Web Application Firewall|WAF]], and [[Network Firewall]]s.

![[Pasted image 20260501143823.png]]

Mental model:
```
(Network) Firewall = locked doors.
WAF = ID check at the door for web visitors.
IDS = security cameras and motion sensors throughout the building, with someone watching the feeds.
IPS = a guard at the choke points who can intervene in real time.
```

