---
aliases:
  - LAN
---
A local Area Network is the network inside a home, office, data center rack, campus building, etc. Devices on the same LAN can talk to eachother directly without going through the public internet.

 
# LAN Networking
- [[Ethernet]]/[[Wi-Fi]]: The local link technologies; Ethernet uses cables, while Wi-Fi uses radio.
- [[Media Access Control|MAC Address]]: Every network interface has a local hardware address like `a4:83:e7:10:9b:2c`
	- [[Switch]]es use the MAC addresses to move traffic around inside a LAN
- [[Internet Protocol|IP Address]]: Devices also get IP addresses, usually private ones like `192.168.1.25` or `10.0.0.42`
	- Applications usually think in terms of IP addresses, not MAC addresses
- A [[Switch]] connects devices on the same LAN. It learns which MAC addresses is reachable on which port, then forwards traffic only where it needs to go.
- [[Routing|Router]]s connect your LAN to other networks, usually the internet. If your laptop wants to reach something outside the LAN, it sends traffic to the router.
- [[Dynamic Host Configuration Protocol]] (DHCP) automatically gives devices network settings. Without it, you would need to configure those manually:
	- [[Internet Protocol|IP Address]]: `192.168.1.25`
	- [[Subnet Mask]]: `255.255.255.0`
	- Default Gateway: `192.168.1.1`
	- DNS server: `192.168.1.1`

# How local communication works:
Scenario: Laptop talking to printer on the same LAN

Assume:
```
Laptop IP:   192.168.1.25
Printer IP:  192.168.1.50
Subnet:      192.168.1.0/24
Router:      192.168.1.1
```

#### 1) The device gets network settings
- When the laptop joins [[Wi-Fi]] or plugs into into [[Ethernet]], it needs local network configuration.
Usually [[Dynamic Host Configuration Protocol|DHCP]] gives it:
```
IP address:      192.168.1.25
Subnet mask:     255.255.255.0
Default gateway: 192.168.1.1
DNS server:      192.168.1.1 or another DNS resolver
```
The [[Subnet Mask]] tells the laptop what other IP addresses count as local.
- With `192.168.1.0/24`, local means: `192.168.1.0` - `192.168.1.255`

So we know that the print's IP `192.1.68.1.50` is a local address.

#### 2) The app chooses a destination
Maybe you open `http://192.168.1.50` , or `http://printer.local`. 
- If you use a name, the laptop first resolves it to an IP address, using [[Domain Name Service|DNS]], [[Multicast DNS|mDNS]], or another local discovery method.

Eventually, the OS has a destination IP: `192.158.1.50`


#### 3) The OS Checks: Local or Remote?
- The laptop compares the destination IP against its subnet.
	- Destination: `192.168.1.50`
	- My subnet: `192.168.1.0/24`

Send directly to the printer.
- If the destination were `8.8.8.8`, then the laptop would decide: "Not local, send the default gateway/router instead."

#### 4) IP Needs a Local Carrier
- Ethernet/WiFi doesn't deliver raw IP packets by IP address -- the local link delivers frames by [[Media Access Control|MAC Address]].
- So the laptop needs this:
	- What MAC address owns `192.168.1.50`?

#### 5) [[Address Resolution Protocol]] (ARP) fines the Printer MAC Adress
- For IPv4, the laptop uses ARP.
- It broadcasts to the local network:
	- Who has `192.168.1.50`? Tell me at `192.168.1.25`
	- Broadcast means that the frame is sent to `ff:ff:ff:ff:ff:ff`
		- Every device on that local broadcast domain receives it, but only the printer should answer:
			- "Hey, I'm the printer, `192.168.1.50` is at `30:9c:23:aa:10:44`!"
- The laptop then stores that in its [[Address Resolution Protocol|ARP]] cache: `192.168.1.50 -> 30:9c:23:aa:10:44`


#### 6) The laptop then builds the real frame
- Now the laptop can send the actual data.
Conceptually:
```
Application Data
	Inside TCP or UDP
		Inside IP packet
			Inside Ethernet/Wi-Fi Frame
```

#### 7) The Switch or Wi-Fi [[Wireless Access Point|AP]] Forwards It
- If this is Ethernet, the switch looks at the destination MAC: `30:9c:23:aa:10:44`
- The [[Switch]] keeps a table like:
```
Laptop MAC  -> port 3
Printer MAC -> port 7
Router MAC  -> port 1
```
- So it forwards the frame only to the printer's port.
- If this is Wi-Fi, the details are slightly different, but the idea is the same: local frames move by MAC address.

### 8) The printer receives and unwraps it
- The printer's [[Network Interface Card]] sees: 
```
To MAC: 30:9c:23:aa:10:44
```
That is its MAC, so it accepts the frame.
Then it unwraps the payload:
```
Ethernet/Wi-Fi frame
    -> IP packet
      -> TCP/UDP
        -> application data
```

#### 9) Reply goes back the same way
The printer replies:
```
From IP: 192.168.1.50
To IP:   192.168.1.25
```
It checks that `192.168.1.25` is local, finds or learns the laptop's [[Media Access Control|MAC]] address, wraps the reply in an (e.g.) Ethernet frame, and sends it back.


# Short Version:
1. App chooses destination IP
2. OS checks whether the destination is on the local subnet (using subnet mask information from DHCP)
3. If local, OS needs the destination's MAC address
4. Check ARP cache or do a broadcast; ARP maps IP -> MAC
5. OS wraps the IP packet inside an Ethernet/Wi-Fi frame.
6. Switch/AP forwards the frame by MAC *address*
7. Destination unwraps the frame and processes the IP packet.
8. Replies follow the same process in reverse.

The router is usually not involved when two devices are truly on the same local subnet.
It becomes involved when the destination is outside the local subnet.