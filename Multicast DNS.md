---
aliases:
  - mDNS
---
A local-network name lookup protocol that lets devices find each other by name without using a normal DNS server.
 
>"DNS for the local LAN." Instead of asking a DNS server, your device asks nearby devices directly: Who is printer.local? And then the matching device answers.

Commonly used for names ending in `.local`, such as:
- `printer.local`
- `macbook.local`
- `living-room-tv.local`

Instead of asking a [[Domain Name Service|DNS]] resolver like `8.8.8.8`, your device asks the local LAN:
- "Who has `printer.local`?"
- That question is sent as a multicast packet to the local network. 
- Devices listening for mDNS receive it.
- If the printer owns that name, it replies: `printer.local is 192.168.1.50`
- Then your laptop can connect to `192.168.1.50`


Normal DNS depends on configured DNS servers and registered zones.
Home and office LANs often have devices that should be discoverable without manual DNS setup:
- Printers
- Smart TVs
- Speakers
- NAS boxes
- Other laptops
- IoT devices

mDNS lets these devices advertise and resolve names locally.

