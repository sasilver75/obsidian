[[AI Engineering]] is the practice of building software products around foundation models.

It sits somewhere between [[Backend Development]], [[Machine Learning]], product engineering, data engineering, and security. The job is usually not "train a model." The job is:

```
User intent -> context -> model/tool calls -> validated output -> product action -> feedback/eval loop
```

The model is one component in a system. The system is the product.

Big idea: ==modern AI engineering is software engineering for probabilistic components.== You still need normal engineering discipline, but tests become evals, logs become traces, API calls become model/tool steps, and product design has to assume partial failure.


# What Changed

From ~2022-2023, "AI app" often meant:

```
prompt -> LLM -> text response
```

The modern stack is more like:

```
user/event
  -> policy + routing
  -> context assembly
  -> model call(s)
  -> retrieval/tool calls/code execution
  -> structured output validation
  -> deterministic business logic
  -> trace/eval/feedback loop
```

The important shift is from ==prompting a model== to ==orchestrating a system of models, tools, data, state, and humans.==


# AI Engineer vs ML Engineer

Roughly:

- [[Machine Learning Engineer]]: trains, fine-tunes, evaluates, serves, and monitors models.
- [[MLOps]] engineer: builds the pipelines and platforms around model lifecycle.
- AI engineer: uses foundation models as infrastructure to build product features.

An AI engineer should be fluent in:
- Backend engineering
- Product UX
- Model APIs
- Prompting and structured outputs
- RAG/search
- Tool calling
- Evals
- Observability
- Cost/latency engineering
- Security and guardrails

This is why the role feels amorphous: it is really an integration role around a fast-moving substrate.


# The Modern AI App Stack

```
UX / Product Surface
Backend / State / Auth / Business Logic
Model Gateway / Provider SDKs
Context Layer: RAG, files, memory, user state
Tool Layer: APIs, code, browser, database, MCP
Orchestration: chains, workflows, agents
Evals + Observability
Feedback + Data Flywheel
```

The stack is good when each layer can be inspected and improved independently.


# Model Layer

The default is to use model APIs, not self-host.

Provider categories:
- Frontier APIs: OpenAI, Anthropic, Google Gemini, xAI, etc.
- Open-weight hosted APIs: Together, Fireworks, Groq, Replicate, Hugging Face Inference, Bedrock, Vertex AI, Azure AI Foundry.
- Self-hosted inference: vLLM, SGLang, TGI, Triton, llama.cpp/MLX for local or edge.
- Gateways: LiteLLM, OpenRouter, Vercel AI Gateway, Portkey, Helicone.

What matters when choosing a model:
- Task quality
- Tool calling reliability
- Structured output reliability
- Latency
- Cost per successful task, not cost per token
- Context window and cache behavior
- Multimodal support
- Data retention / privacy terms
- Rate limits
- Model version stability

Rule: ==pin model versions when you can.== Model migrations are production migrations.


# Model Gateways

Once you use more than one provider/model, a gateway becomes useful.

Gateway responsibilities:
- Unified API across providers
- Model routing
- Fallbacks
- Rate limiting
- Spend caps
- Prompt/model version tagging
- Logging/tracing
- Redaction
- Tenant/team/customer cost attribution
- Policy enforcement

Tools:
- LiteLLM
- OpenRouter
- Vercel AI Gateway
- Portkey
- Helicone
- Custom internal gateway

Do not over-abstract too early. Provider differences matter: streaming events, tool calling, structured outputs, file handling, caching, and safety behavior are not perfectly portable.


# API Primitives

Modern model APIs converged around a few primitives:

- Messages: role-tagged conversation state.
- System/developer instructions: durable behavioral constraints.
- Structured outputs: JSON-schema-ish output contracts.
- Tool/function calling: model chooses a typed action; your code executes it.
- Built-in tools: web search, file search, code execution, image generation, computer use, etc.
- Streaming: token/event streams for responsiveness.
- Reasoning/trace items: provider-managed intermediate state in some APIs.
- Prompt/context caching: cheaper/faster repeated prefixes.
- Batch APIs: async cheaper processing for non-interactive workloads.

OpenAI-specific note: new work should generally use the [[Responses API]] rather than the older Assistants API. The broader pattern matters more than the vendor name: one API call may now include text, images, tool calls, web/file search, code execution, and traceable state.


# Structured I/O

For production AI apps, free-form text is the least useful output format.

Prefer:

```json
{
  "decision": "approve",
  "confidence": 0.82,
  "reasons": ["income verified", "low debt ratio"],
  "needs_human_review": false
}
```

Instead of:

```
Looks good to me.
```

Why:
- Easier validation
- Easier retries
- Easier evals
- Safer downstream automation
- Better logs and analytics

Tools/patterns:
- Provider structured outputs
- Pydantic / Zod schemas
- Instructor
- Outlines / Guidance / constrained decoding
- JSON repair only as a fallback

Rule: ==validate model output like untrusted user input.== The schema is a contract, not a guarantee of semantic correctness.


# Prompting

Prompting is still useful, but it is no longer the whole field.

Good prompts:
- Are short and specific.
- Separate task, context, constraints, examples, and output schema.
- Use examples that match production inputs.
- Tell the model what to do when context is insufficient.
- Avoid mixing many unrelated jobs in one prompt.
- Put volatile context outside the stable instruction block.
- Are versioned and evaluated.

Bad prompts:
- Become 4,000-token policy blobs nobody can reason about.
- Hide business logic that should be code.
- Depend on undocumented model quirks.
- Treat "think step by step" as magic.

Useful pattern:

```
classify -> retrieve -> extract -> verify -> write
```

instead of one giant "do everything" prompt.


# Context Engineering

[[Context Engineering]] is the art of deciding what the model sees.

Context can include:
- User profile and permissions
- Conversation history
- Retrieved documents
- Tool results
- Product state
- Examples
- Policies
- Output schema
- Scratchpad / plan / task state

The hard part is not "put more in the context window." The hard part is putting the right information in the right structure at the right time.

Modern long-context models are useful, but they do not remove the need for retrieval, compression, ranking, access control, and freshness.

Rule: ==context is a scarce resource even when the context window is huge.==


# RAG

[[Retrieval-Augmented Generation|RAG]] is the default way to add external knowledge.

The naive version:

```
chunk docs -> embed chunks -> vector search -> stuff top-k into prompt
```

The production version:

```
ingest -> parse -> clean -> chunk by document structure
      -> embed + keyword index + metadata
      -> retrieve -> rerank -> filter by permissions/freshness
      -> cite -> answer -> evaluate retrieval and generation separately
```

Things that matter more than people expect:
- Document parsing quality
- Chunking strategy
- Metadata
- Access control
- Hybrid search: [[BM25]] + vectors
- Query rewriting
- Reranking
- Deduplication
- Freshness / invalidation
- Citations
- Evaluation of retrieval separately from answer quality

Vector search is not magic. For many production queries, keyword search is still the baseline to beat.

Tools:
- LlamaIndex: data/RAG framework
- LangChain: broad LLM app framework
- Qdrant, Weaviate, Pinecone, Milvus, Chroma, LanceDB
- [[pgvector]] when Postgres is already enough
- Elasticsearch/OpenSearch/Typesense/Meilisearch for keyword/search-heavy systems
- Cohere/Voyage/Jina/etc. for embeddings/reranking


# Tool Calling

Tool calling turns language into typed action.

Examples:
- Search docs
- Query a database
- Send email
- Create a Linear ticket
- Read a file
- Run code
- Fetch a URL
- Update a CRM record

The model should decide *whether* to call a tool and with *what arguments*. Your code should decide:
- Whether the tool is allowed
- Whether arguments are valid
- Whether the user has permission
- Whether the action is safe
- How to retry/fail
- How to audit it

Tool descriptions are part of the prompt surface. Bad tool docs cause bad tool use.

Rule: ==tools should be narrow, typed, idempotent where possible, and observable.==


# MCP

[[Model Context Protocol]] is becoming the default vocabulary for connecting AI systems to tools and data sources.

Mental model:

```
AI client/host <-> MCP server <-> tool/data system
```

Why it matters:
- Standardizes tool discovery and invocation.
- Lets tools be reused across clients.
- Moves integration work out of one-off glue code.
- Gives AI IDEs/agents a shared way to connect to files, databases, SaaS tools, and internal APIs.

Risk:
- MCP is a capability boundary. Treat every MCP server like a plugin with permissions, secrets, and potential prompt-injection exposure.
- Avoid giving broad filesystem, shell, database, or production-write access without review.
- Prefer explicit allowlists, scoped credentials, sandboxing, logging, and user approval for sensitive actions.

MCP is useful, but it is not a security model by itself.


# Agents

An [[Agent]] is a system that can choose actions over multiple steps to accomplish a goal.

Minimal loop:

```
observe -> think/plan -> act -> observe -> ...
```

In practice, reliable agents look less like an unconstrained autonomous being and more like a workflow with model-driven parts.

Agent architecture options:
- Single model call: simplest, often enough.
- Chain: fixed sequence of model/tool steps.
- Router: classify task -> send to specialized path.
- DAG/workflow: explicit graph with branches.
- State machine: controlled transitions, safer for production.
- Agent loop: model decides next step dynamically.
- Multi-agent: usually overused; helpful only when roles/tools/state are genuinely separable.

Tools/frameworks:
- LangGraph: durable graph/state-machine style agents.
- LlamaIndex Workflows: RAG/agent/data workflows.
- OpenAI Agents SDK: provider-native agent runtime/tracing.
- Pydantic AI: Python typed agents.
- Mastra / Vercel AI SDK: TypeScript AI app/agent stack.
- CrewAI/AutoGen/etc.: multi-agent experimentation, with mixed production fit.

Rule: ==start deterministic, add agency only where it buys something.==


# Workflows Beat Vibes

The most reliable "agentic" systems often look like:

```
1. classify the request
2. retrieve relevant context
3. call exactly one allowed tool
4. validate output
5. ask human if confidence/risk threshold is crossed
6. write result
```

Not:

```
You are an autonomous agent. Figure it out.
```

The failure rate compounds across steps. If a task needs 12 successful model decisions in a row, it is much harder than it looks.

Use explicit workflows for:
- Compliance-sensitive domains
- Money movement
- Customer-visible actions
- Production writes
- Long-running tasks
- Anything with real-world consequences


# Durable Execution

Many AI tasks should not run inside a single HTTP request.

Examples:
- Research over many sources
- Document extraction over large files
- Multi-step agent runs
- Batch enrichment
- Human approval flows
- Long tool chains with retries

Use background jobs, queues, or [[Durable Execution Engine]]s so work can resume after crashes and retries do not duplicate side effects.

Patterns:
- Persist run state.
- Make tool calls idempotent.
- Add explicit timeouts.
- Store intermediate artifacts.
- Let humans resume/approve/reject.
- Expose run status in the UI.
- Keep a full trace of the run.

Tools:
- [[Temporal]]
- Restate
- Inngest
- BullMQ/Celery/RQ for simpler queues
- LangGraph durable execution for agent state
- Cloud workflow engines where appropriate


# Memory

"Memory" means several different things:

- Conversation history: recent messages.
- User profile: durable preferences/facts.
- Episodic memory: past interactions.
- Semantic memory: searchable facts/notes.
- Task state: what the agent has done so far.
- Product state: records in the real application database.

Do not dump all memory into the prompt.

Better:
- Store memory as normal application data.
- Retrieve only relevant memory.
- Track provenance and timestamps.
- Let users inspect/edit/delete durable memory.
- Treat memory writes as product events, not model side effects.

Rule: ==memory is a database problem before it is a prompt problem.==


# Evals

Evals are the test suite for AI systems.

Types:
- Unit evals: small deterministic checks.
- Golden set evals: representative real examples.
- Regression evals: known failures never return.
- Retrieval evals: did we fetch the right evidence?
- Tool evals: did the model call the right tool with valid args?
- Structured-output evals: did it match the schema and semantics?
- Human evals: needed for subjective/high-risk behavior.
- Production evals: sampled traces scored after deployment.

Useful metrics:
- Task success rate
- Refusal/abstention correctness
- Groundedness / citation correctness
- Retrieval precision/recall/NDCG
- Tool-call accuracy
- Hallucination rate
- Latency
- Cost per successful task
- Human correction rate

Rule: ==do not ship serious AI features without a small eval set.== It can start ugly. It cannot stay absent.


# Observability

Traditional logs answer:

```
Did the request fail?
```

AI traces answer:

```
What did the model see?
What did it retrieve?
What tools did it call?
What did it output?
What did that cost?
Where did quality degrade?
```

Trace the full path:
- Input
- Prompt version
- Model and parameters
- Retrieved docs
- Tool calls and outputs
- Structured output
- Latency
- Token usage/cost
- User feedback
- Eval scores

Tools:
- [[Langfuse]]
- [[LangSmith]]
- [[Weave]]
- Braintrust
- Arize Phoenix
- Helicone / Portkey
- OpenTelemetry where possible

Rule: ==if you cannot inspect bad outputs, you cannot improve the system.==


# Guardrails

Guardrails are runtime constraints. Evals measure quality; guardrails prevent or route failures.

Common guardrails:
- Input moderation
- Output moderation
- Prompt-injection detection
- Tool permission checks
- PII redaction
- Schema validation
- Citation requirement
- Confidence thresholds
- Human approval gates
- Rate limits and spend caps
- Action allowlists

Guardrails should be layered:

```
before model -> during tool/action -> after model -> before user/action
```

Do not rely on a single "safety prompt" to protect production systems.


# Security

AI systems expand the attack surface.

New-ish risks:
- Prompt injection
- Data exfiltration through tools
- Indirect prompt injection from retrieved docs/webpages/emails
- Overbroad agent permissions
- Tool argument injection
- Cross-tenant data leakage in RAG
- Model output treated as trusted code/SQL/HTML
- Leaked prompts, traces, files, or secrets
- Malicious MCP/tool servers

Basic discipline:
- Run tools with least privilege.
- Separate read tools from write tools.
- Require human approval for high-impact actions.
- Sanitize model-generated HTML/Markdown.
- Parameterize SQL.
- Do not put secrets in prompts.
- Log tool use.
- Respect tenant/user authorization at retrieval time, not only at display time.
- Sandbox code/browser/computer-use tools.

Rule: ==the model is not inside your trust boundary.==


# UX Patterns

AI product UX is not just chat.

Useful patterns:
- Autocomplete / copilot suggestions
- Draft generation with edit/accept
- Extraction into structured fields
- Review queues
- Inline explanations
- Search + answer with citations
- Side panel assistant
- Batch enrichment
- Voice interface
- Agent-run status view
- Human approval checkpoints

Defensive UX matters:
- Show sources/citations when grounding matters.
- Let users inspect/edit before irreversible actions.
- Expose uncertainty honestly.
- Make feedback cheap.
- Prefer "draft" over "send" for risky outputs.
- Avoid making the user reverse-engineer what happened.

The best AI products often make the human faster while keeping them in control.


# Cost and Latency

AI features are unusually sensitive to cost/latency.

Levers:
- Use the smallest model that passes evals.
- Route easy tasks to cheaper/faster models.
- Cache stable prompt prefixes.
- Cache deterministic outputs.
- Use batch APIs for offline work.
- Stream interactive responses.
- Split work into cheap classifier/router + expensive generator.
- Avoid stuffing huge context by default.
- Summarize/compress long histories.
- Parallelize independent tool calls.
- Stop generation early when possible.
- Track cost per successful task, not cost per request.

Model routing is becoming a normal backend pattern:

```
cheap model -> if uncertain/escalated -> stronger model -> if risky -> human
```


# Multimodal and Voice

AI engineering is no longer text-only.

Modalities:
- Text
- Images
- Audio
- Video
- Documents/PDFs
- Screenshots
- UI/computer control

Voice agents are especially different:
- Latency is product quality.
- Turn-taking matters.
- Interruptions/barge-in matter.
- Streaming matters.
- Tool calls must not create awkward dead air.
- Transcripts become eval/observability artifacts.

Realtime APIs and live multimodal APIs make this easier, but product quality still depends on conversation design, latency budgets, and graceful failure.


# Fine-Tuning

Use fine-tuning when:
- You have many high-quality examples.
- Prompting/RAG/evals show a stable gap.
- The task is repetitive and narrow.
- You need lower latency/cost with a smaller model.
- You need style/format consistency that prompting cannot reliably produce.

Do not fine-tune for:
- New factual knowledge. Use RAG.
- A vague product idea.
- Avoiding evals.
- Making a bad workflow good.

Modern order of operations:

```
prompting -> structured output -> RAG/tools -> evals -> collect examples -> fine-tune
```

Fine-tuning is optimization, not product discovery.


# Synthetic Data

Synthetic data is useful when grounded by real data and evals.

Good uses:
- Generate edge-case evals.
- Expand examples for classification/extraction.
- Create adversarial prompts.
- Produce tool-call traces for testing.
- Distill behavior from a stronger model into a cheaper one.

Bad uses:
- Filling a dataset with unverified model guesses.
- Training on synthetic outputs with no quality filter.
- Mistaking volume for signal.

Rule: ==synthetic data needs curation, provenance, and evals.==


# Feedback Loops

The production data flywheel is the real compounding asset.

Capture:
- User thumbs up/down
- Edits to generated drafts
- Regenerations
- Abandoned outputs
- Human reviewer decisions
- Support escalations
- Tool-call failures
- Search queries with no good answer
- Documents users manually attach after a bad answer

Turn those into:
- New eval cases
- Better prompts
- Better retrieval metadata
- Better chunking/parsing
- Fine-tuning examples
- Product requirements
- Guardrail rules

Rule: ==every production failure should have a path to becoming an eval.==


# AI-Native Development

AI engineering also changes how software gets built.

Modern developer loop:
- Use coding agents for scaffolding, refactors, tests, search, and migration work.
- Keep plans and constraints explicit.
- Review generated code like junior engineer output with weirdly high typing speed.
- Prefer small tasks with clear acceptance criteria.
- Add tests before letting agents change broad surfaces.
- Capture decisions in docs so future agents have context.

Coding agents compress implementation time, which makes planning and review more important, not less.


# Small-Team Defaults

For a first useful AI product:

```
TypeScript or Python app
OpenAI/Anthropic/Gemini API
Zod or Pydantic schemas
Postgres + pgvector OR a hosted vector DB
Hybrid search if docs matter
Langfuse/LangSmith/Braintrust for traces/evals
GitHub Actions eval run
Human feedback buttons
Cost/latency dashboard
```

Avoid:
- Fine-tuning before you have examples.
- Multi-agent systems before one-agent works.
- Self-hosting GPUs before unit economics force it.
- Giant prompts with hidden business logic.
- RAG without retrieval evals.
- Tool access without permissions and audit logs.


# Enterprise Defaults

For a serious internal/customer-facing platform:

```
Model gateway
Provider abstraction
Central prompt/schema registry
RAG ingestion pipeline
Tenant-aware retrieval
MCP/tool registry with permissions
Evals in CI
Trace store + production sampling
Human review queues
Security review for tools/actions
Cost attribution by feature/team/customer
```

The platform should make the safe path the easy path:
- approved models
- approved tools
- standard tracing
- standard eval harness
- standard prompt/version deployment
- standard redaction and retention


# Trends to Internalize

1. ==Evals are the new tests.== They are the difference between engineering and demoing.
2. ==Agents are workflows with model-driven control points.== Treat open-ended autonomy as a risk budget.
3. ==RAG is mostly data engineering and search engineering.== The vector DB is the easy part.
4. ==Structured outputs moved AI closer to normal software.== But validation still matters.
5. ==MCP is becoming the USB-C for AI tools.== Useful standard, not automatically safe.
6. ==Model routing is normal.== Use cheap/fast models until the task requires expensive reasoning.
7. ==Voice and multimodal are first-class.== AI apps are moving beyond chat boxes.
8. ==Observability and evals are converging.== Production traces become eval cases; eval failures become product work.
9. ==The moat is the feedback loop.== Prompts and models are easy to copy; production data and domain evals are not.
10. ==The product is the system around the model.==


# Vocabulary

- [[Prompt Engineering]]: crafting instructions/examples/context for a model.
- [[Context Engineering]]: selecting and structuring everything the model sees.
- [[Retrieval-Augmented Generation|RAG]]: retrieving external context before generation.
- [[Tool Calling]]: model emits a typed request for code/API execution.
- [[Structured Outputs]]: model responses constrained to a schema.
- [[Agent]]: AI system that can choose actions over multiple steps.
- [[Model Context Protocol|MCP]]: protocol for exposing tools/data to AI clients.
- [[LLM-as-a-Judge]]: using an LLM to score/compare outputs.
- [[Prompt Injection]]: malicious or accidental instructions in user/retrieved content that hijack behavior.
- [[Semantic Cache]]: cache keyed by meaning rather than exact string.
- [[Data Flywheel]]: production usage creates examples/feedback that improve the system.


# Mental Model

Modern AI engineering is a set of loops:

```
Build feature
Instrument traces
Review failures
Write evals
Improve prompt/context/tools/workflow
Ship behind guardrails
Collect feedback
Repeat
```

If the loop is tight, the system improves. If the loop is absent, the product runs on vibes.
