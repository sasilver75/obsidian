The active execution environment that sits between your program and the underlying machine or platform, making it possible for the program to run in a predictable way.

The layer of software that provides the conditions a program needs in order to execute. Programs are often not run by the hardware directly, it needs some surrounding environment:
- Memory management
- Access to files and network
- Libraries
- System calls
- Scheduling
- Error handling
- Configuration
- Interpretation or Just-in-Time Compilation
The runtime is the part of the system that supplies or coordinates these things while the program is running.

Answers:
- How doest the program start?
- What code is available by default?
- How does it access the operating sytem?
- How are memory, threads, errors, and I/O handled?
- What rules exist while the program is executing?


==The exact meaning depends on context. Examples:==
- In [[Container]]s, a container runtime is the component that starts, runs, stops, and manages containers on a machine.
- A serverless runtime loads your function, provides the request/response interface, handles environment variables, and defines what language versions and libraries are available.
- For a compiled language like C, the runtime might be relatively small: Startup code, standard libraries, memory allocation, thread support, and operating system integration
- For a managed language like Java, Python, or JavaScript, the runtime is much more visible. It might include an interpreter or virtual machine, garbage collector, module loader, standard library, event loop, security model, and tooling hooks.
	- The Java Runtime Environment runs Java bytecode
	- [[Node.js]] runs JavaScript outside the browser
	- [[CPython]] runs Python programs