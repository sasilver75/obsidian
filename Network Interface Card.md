---
aliases:
  - NIC
  - Network Card
---
The hardware component that connects a device to a network. The thing that actually sends and receives network signals. A device's network adapter, the local-network doorway between the computer and the network.

Examples:
- [[Ethernet]] port/card
- [[Wi-Fi]] adapter
- USB-to-Ethernet adapter
- Cellular modem

A [[Network Interface Card|NIC]] typically has a [[Media Access Control|MAC Address]] such as `a4:83:e7:9b:2c`

That MAC address identifies that network interface on the local network. In a laptop, you might have multiple NICs:
```
Wi-Fi interface:      en0
Ethernet adapter:     en5
Bluetooth PAN:        en7
VPN virtual adapter:  utun4
```
Each interface can have its own MAC address, IP address, routing behavior, and link status.
The NIC handles the low-level work of moving frames over the physical or wireless medium:
- Ethernet NIC -> Electrical/optical signals over cable
- Wi-Fi NIC -> Radio signals over air

