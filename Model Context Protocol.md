References
- Latent Space: [The Creators of Model Context Protocol](https://youtu.be/m2VqaNKstGc?si=HTKThmII-xH6Mvf1)
- Docs: [Model Context Protocol](https://modelcontextprotocol.io/introduction)
- 

A protocol designed to help ==AI applications== (not models themselves) extend themselves and integrate with an ecosystem of plugins.
MCP is kind of like the USB-C port of AI applications, in that it's meant to be a universal connection to a whole ecosystem of datasources/tools.
There's a Client and Server nomenclature to it.

MCP provides:
- A growing list of pre-built integrations that your LLM can directly plug into.
- The flexibility to switch between LLM providers and vendors.
- Best practices for securing your data within your infrastructure.

![[Pasted image 20250407174420.png|600]]

- ==MCP Hosts==: Programs like Claude Desktop, IDEs, or AI tools that want to access data through MCP.
- ==MCP Clients==: Protocol clients that maintain a 1:1 connection with servers.
- ==MCP Servers==: Lightweight programs that each expose specific capabilities through the standardized Model Context Protocol.
- Local Data Sources: Your computer's files, databases, and services that MCP servers can securely access.
- Remote Services: External systems available over the internet (e.g. through APIs) that MCP servers can connect to.

Core architecture:
- MCP follows a Client-Server Architecture where:
	- ==Hosts== are LLM applications (Claude Desktop, IDEs) that initiate connections
	- ==Clients== maintain 1:1 connections with servers, *inside* the host application
	- ==Servers== provide context, tools, and prompts to clients
![[Pasted image 20250407174752.png|400]]

Protocol Layer
- Handles message framing, request/response linking, and high-level communication patterns.
- Key classes include Protocol, Client, Server

Transport Layer
- Handles the actual communication between clients and servers. MCP supports multiple transport mechanisms:
	- Stdio transport
		- Uses standard Input/output for communication. Ideal for local processes.
	- HTTP with SSE transport
		- Uses [[Server-Sent Event]]s (SSE) for server-to-client messages.
		- HTTP POST for client-to-server messages
- All transports use JSON-RPC 2.0 to exchange messages. See the specification for detailed information about the Model Context Protocol message format.

MCP has three main types of messages:
1. ==Requests== expect a response from the other side
2. ==Results== are successful responses to requests
3. ==Errors== indicate that a request failed
4. ==Notifications== are one-way messages that don't expect a response

Connection Lifecycle:
![[Pasted image 20250407175101.png|500]]
- Initialization
	- Client sends initialize request with protocol version and capabilitites
	- Server responses with its protocol version and capabilities
	- Client sends initialized notification as ack
	- Normal message exchange begins
- Message exchange: The following patterns are supported
	- Request-Response: Client or server sends requests, the other responds.
	- Notifications: Either party sending one-way messages
- Termination: Either party can terminate the connection
	- Clean shutdown via close()
	- Transport disconnection
	- Error conditions
- Error handling: MCP defines these standard error codeS:
	- ParseError, InvalidRequest, MethodNotFound, InvalidParams, InternalError
	- SDKs and applications can define their own error codes above -32000

Local communication
- Uses stdio transport for local processes
- Efficient for same-machine configuration
- Simple process management

Remote communication
- Use SSE for scenarios requiring HTTP compatibility
- Consider security implications including authentication and authorization

Message Handling
1. Request processing
	1. Validate inputs thoroughly
	2. Use type-safe schemas
	3. Handle errors gracefully
	4. Implement timeouts
2. Progress reporting
	1. Use progress tokens for long operations
	2. Report progress incrementally
	3. Include total progress when known
3. Error management
	1. Use appropriate error codes
	2. Include helpful error messages
	3. Clean up resources on errors

Security Considerations
1. Transport security
	1. Use [[Transport Layer Security]] for remote connections
	2. Validation connection origins
	3. Implement authentication when needed
2. Message validation
	1. Validate all incoming messages
	2. Sanitize inputs
	3. Check message size limits
	4. Verify JSON-RPC format
3. Resource protection
	1. Implement access controls
	2. Validate resource paths
	3. Monitor resource usage
	4. Rate limit requests
4. Error handling
	1. Don't leak sensitive information
	2. Log security-relevant errors
	3. Implement proper cleanup
	4. Handle DoS scenarios

Debugging and monitoring
1. Logging
	1. Log protocol events
	2. Track message flow
	3. Monitor performance
	4. Record errors
2. Diagnostics
	1. Implement health checks
	2. Monitor connection state
	3. Track resource usage
	4. Profile performance
3. Testing
	1. Test different transports
	2. Verify error handling
	3. Check edge cases
	4. Load test servers


Creators found themselves copy things back between Claude Desktop and their IDE; How do we fix this? 
Clearly seems to be an MxN problem: Multiple applications and multiple integrations; what better way to fix this than a protocol!

> "Fundamentally, every primitive we thought through, we thought form the perspective of the application developer first. What are the different things that I would want to receive from an integration? Through that lens, 'tool calling' is necessary but insufficient. We need to further differentiate... the core primitives... The first are ==tools== (or function calling), the second is ==resources== (basically bits of data or context that you might want to add to the model context)... this is one of the first primitives we thought could be application controlled. You might want the application (or the application UI) to be able to search through and tag certain resources that become part of the message to the LLM. The third one primitive is ==prompts== (meant to be user-initiated, or user-substituted text or messages). This could be a /command or an @command in an editor -- it's a macro, effectively, that a user wants to drop in and use."

> From the application developer perspective, application developers don't want to be commoditized; what are some of the unique things they could do to crate the best user experience?

> To add to that... interestingly, while tool calling is probably 95%+ of the integrations we see... the very first implementation in Zed is actually a prompt implementation that doesn't deal with tools; it lets you build an MCP server that takes a backtrace from Sentry and lets you pull this into the context window beforehand; it's quite nice that it's a *user-driven interaction* where you as a user decide when to pull this in, and don't have to wait for the model to do it. It's a great way to craft the prompt, in a way. 

When should people use a tool versus a resource, especially for things that DO have an API interface? Like for a Database, you can do a tool that does the SQL query, but when should you do this for a resource?
> Tools are always meant to be initiated by the model, at the model's discretion -- it will find the right tool and apply it. If that's the interaction you want as a server developer - that you've given the LLM the ability to run SQL queries -- that makes sense as a tool. But resources are more flexible, basically. The story is practically a bit complicated today because many clients don't support resources yet, but in an ideal world, you would do resources for things like schemas of your database tables, to either allow users to say "Claude, I want to talk to you about database table @blah", or the application you're using, like Claude Code, is able to agentically look up resources and find the right schema of the database that you're talking about. Any time you want to list a bunch of entities and read a bunch of them, that makes sense to model as a resource. Resources are also uniquely identified by a URI.

![[Pasted image 20250407145401.png]]
Swyx: I think this document is so useful that it should be on the front pages of the Docs. As a DevRel person, I always insist of having a map for people: "Here's a map of all the things you should understand; let's spend two hours going over this."

> People still apply their usual SWE API engineering approaches to this, but I think we still need to relearn how to build something for LLMs and trust them, particularly as they're getting significantly better year-to-year. Nowadays, just throw data at the thing that's really good at dealing with data (an LLM). We need to unlearn 20-30 years of SWE practices that go into this, to some degree. 

> One framing for MCP is to think about how crazy-fast AI is advancing; us thinking that the biggest bottleneck to the next wave of capabilities of models might be their ability to interact with the outside world, taking actions, etc. As AI get better, this will be key to becoming productive with AI; so MCP is sort of a bet on the future and this being more important going forward.

> I think more statefulness will become more popular, especially when we talk about modalities that go beyond text-based interaction with models. I think that having something a bit more stateful is going to become more useful... in this interaction pattern. People look for these A vs B scenarios. If you want to have a rich interaction with an AI application, MCP is probably the choice. If you want to have an API spec that a model can read and interpret, the OpenAI spec may be a way to go... People in the community have built bridges as well; if you have an OpenAI spec, there are already translators out there that will take it and re-expose it as MCP, and you can go the other direction too.

Build the servers... people do their tweets about connecting Claude Desktop to x MCP, and it's amazing, but how would you suggest people starting building servers?
- How do you draw the line between being very descriptive vs just "take the data," and let the model manipulate it later.
> With MCP, it's easy to build something that's pretty good in half an hour; so pick the language of your choice that you love the most, pick an SDK for it, and build the tool that matters to you, personally! Build a server, throw the tool in, and don't worry too much about the description just yet -- write your little description as *you* think about it, throw it to the model, standard IO protocol into an application you like, and see it do things. It's empowerment and magic for developers. Adds a lot of fun to the development piece too, to have models that can go out and do this.

 


