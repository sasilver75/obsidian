A number that tells the device which application should handle the data.
- [[Internet Protocol|IP Address]] + [[Media Access Control|MAC Address]] gets you to the right device, and Port gets you to the right application.
	- Classically:
		- Port 443 (TCP) for web traffic
		- Port 9812 (UDP) for your video call

Numbered channels on a server, ranging from 1 to 65535
Different apps listen on different ports.
- 80 is a standard port for web servers
- 443 is a standard port for secure web connections
- MySQL port often listens on 3306

> Think of it as an apartment building; 2436 Main Street is an address ([[Internet Protocol|IP]] Address), while the apartment number is the port.


> Only one listener can normally bind a given (transport protocol + local IP address + port + network namespace) combination at a time. So you *can* have App A listening on [[Transport Control Protocol|TCP]] :3000 and App B listening on [[User Datagram Protocol|UDP]] :3000.



