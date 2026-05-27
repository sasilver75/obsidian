---
aliases:
  - BLE
---
cf [[Bluetooth]] Classic

BLE: Bluetooth Low Energy
- It is a version of [[Bluetooth]] designed for low-power, short-range wireless communication, especially for devices that have to run a long time on small batteries.
- Use cases:
	- Fitness trackers
	- Smartwatches
	- Medical sensors
	- Beacons ((we used for Bird scooters))
	- Asset trackers
	- Smart locks
	- Environmental sensors
- Compared with [[Bluetooth|Bluetooth Classic]]
	- BLE uses less power and is better for small bursts of data.
	- Bluetooth Classic is better for continuous high-bandwidth use like traditional audio streaming.


Common BLE Concepts
- Peripheral: The device advertising a service, such as a sensor.
- Central: The device that scans and connects, such as a phone.
- Advertisement: A short broadcast packet used for discovery.
- Service: A group of related functions or data.
- Characteristic: A specific value that can be read/written/notified.
- GATT: Generic Attribute Profile: The structure BLE uses for services and characteristics.

