---
aliases:
  - A2A
---
April 9, 2025
[Google Blog Post](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)

An open ==protocol== called A2A from Google, with support for contributions from more than 50 technology partners. The A2A protocol will allow AI agents to communicate with eachother, securely exchange information, and coordinate actions on top of various enterprise platforms or applications. 
- Complements [[Model Context Protocol]] (MCP), which provides helpful tools and contexts to agents.

> "To maximize benefits from agentic AI, it is critical for these agents to be able to collaborate in a dynamic, multi-agent ecosystem across siloed data systems and applications, even if they were built by different vendors or in a different framework."

# Design Principles
- Embrace agentic capabilities, enabling agents to collaborate in their natural, unstructured modalities.
- Build on existing standards: The protocol is build on top of existing, popular standards, including [[HTTP]], [[Server-Sent Event|SSE]], [[JSON-RPC]], meaning its easy to integrate.
- Designed to support enterprise-grade [[Authentication|Authn]] and [[Authorization|Authz]]
- Support for long-running tasks: We designed A2A to be flexible and support scenarios where it excels at completing everything from quick tasks to deep research that may take hours or even days, when humans are in the loop.






