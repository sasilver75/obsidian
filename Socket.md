
An [[Operating System|OS]] object that a program uses to send and receive data.

In entworking, it's the program's handle to a network communication path.

```
application code
    |
    | read() / write() / send() / recv()
    v
  socket object in the OS kernel
    |
    v
  TCP/IP stack
    |
    v
  network card / network
```

The application itself doesn't manipulate raw network packets directly, it asks the OS to read from or write to a socekt.


# Comparison with [[Port]]
- A port is just a number, like 443.
- A socket is an actual OS-managed communication object.

For a [[Transport Control Protocol|TCP]] connection, a connected socket is usually identified by:
```
protocol: TCP
local IP: 10.0.1.5
local port: 443
remote IP: 203.0.113.9
remote port: 53144
```
Together, this tuple identifies one TCP connection.

So a server can have many sockets that use the same local port `443` at the same time, because each client has a different remote IP/port combination. Same server port, different sockets.


# Two types of Sockets
On a server, there are two important types of TCP sockets:
1. ==Listening Socket==: `listen on 0.0.0.0:443`, the socket waits for a new connection. When a client connects, the OS creates a separate Connected Socket for that specific client.
2. ==Connected Socket==: The OS object representing one specific live network connection between two endpoints. The server can read from it `recv(socket)` and write to it `send(socket, bytes)`.


