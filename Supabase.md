
An open-source [[Backend-as-a-Service]] that bundles a managed [[PostgreSQL|Postgres]] database with a suite of complementary services.

It's often positioned as the open-source alternative to [[Firebase]], but it's architecturally quite different: Supabaes is "Postgres + tools around it," while Firebase is built on Google's proprietary document store. Supabase is self-hostable: every component is OSS, so you can run the whole stack on your own infra if you ever needed to leave.

# Core Features

### Postgres Database
- Managed Postgres instance with extensions enabled, including:
	- [[PostGIS]] for geospatial queries
	- [[pg_cron]] for scheduled jobs running inside the database
	- [[pgvector]] for vector embeddings for [[Vector Search|Semantic Search]]
	- [[pg_net]] for making HTTP calls from inside Postgres (useful for webhooks fired by triggers)
	- [[pgmq]] as a lightweight message queue
	- [[pg_graphql]] for an auto-generated GraphQL API from your schema.
- You get a real Postgres `DATABASE_URL` that you can connect to with any standard driver (pgx, psycopg, etc.). 
- You can run migrations via Supabase CLI, golang-migrate, goose, etc.

### Authn
- A managed authentication service supporting:
	- Email + Password
	- Magic links (via email)
	- Phone OTP (via Twilio, MessageBird, Vonage, or AWS SNS)
	- [[OAuth]] providers (Google, Apple, GitHub, Discord, ~30 others)
	- [[Security Assertion Markdown Language|SAML]]/[[Single Sign-On|SSO]] (paid)
	- Anonymous auth
- Issues [[JSON Web Token|JWT]]s that other Supabase services (and your own backend services) can validate via published [[JSON Web Key Set|JWKS]]. Auth state lives in an `auth.users` table in the same Postgres database, so user IDs are foreign-keyable from your tables!

### Supabase Realtime
- [[WebSockets|WebSocket]] layer with three capabilities:
	- Postgres Changes: Subscribe to INSERT/UDPATE/DELETE on specific tables with filters; rows diffs streams via [[Logical Replication]].
	- Broadcast: Ephemeral [[Publish-Subscribe|Pub Sub]] messaging that doesn't touch the database,
	- Presence: Track which users are currently connected to a channel.
- Authorization respects [[Row-Level Security]], so subscribers only receive row changes they'd be allowed to SELECT.

#### Storage
- [[Amazon S3|S3]]-compatible object storage for files (images, videos, documents)
- Features
	- Bucket-level policies
	- Image transformations on the fly (resize, quality, format)
	- Resumable uploads
	- CDN delivery

#### Edge Functions
- TypeScript functions running on [[Deno]], deployed globally at the edge (people hate Deno it seems)
- Used for:
	- [[Webhook]] handlers
	- Third-party API calls
	- Custom server-side logic that doesn't fit in the database itself
	- Per-invocation; no long-running workers (~400s wall-clock limit)

#### [[PostgREST]]
- Auto-generated [[Representational State Transfer|REST]] API derived from your PG schema
- Tables, views, and functions become HTTP endpoints automatically. 
- Combined with [[Row-Level Security|RLS]], your iOS app can hit Postgres directly without you writing CRUD endpoints.


#### Row Level Security ([[Row-Level Security|RLS]])
- Postgres-native feature, but central to the Supabase model.
- Policies enforce per-row access rules in the database itself, ==making it safe for clients to talk to Postgres directly.==

#### Vault
- Encrypted secrets storage inside Postgres for API keys, tokens, etc. Backed by `pgsodium`


### Cron ([[pg_cron]])
- Scheduled jobs that run as SQL or call Edge Functions/webhooks.
- Used for things like: "Every minute, find Filling events past their tip-deadline and cancel them!"


### [[Webhook]] (Database Webhooks)
- Configure [[HTTP]] calls to fire automatically when rows change in specified tables... Built on [[pg_net]]. 
- Useful for: "When an event Tips, ping the Go server to fan out push notifications."


#### Studio
- A web-baesd admin UI for browsing and editing data, writing SQL, managing auth users, configuring storage, viewing logs. An admin and monitoring surface.

#### CLI
- `supabase` CLI for local development (spins up the entire stack in Docker), running migratinos, generating typed clients, deploying Edge Functions.

#### AI/Vector
- first class [[pgvector]] support with helpers for embedding storage and similarity search.

#### Logs and Observability
- Built-in log explorer for Postgres, Auth, Realtime, Storage, Edge Functions, API requests, etc. Backed by LogFlare (acq. Supabase)

#### Client SDKs
- Official SDKs for JS/TS, Swift, Kotlin, Flutter, Python, and others.

#### Pricing model
- Free tier is generous for small projects; limits on DB size (500MB), bandwidth, monthly active users (50k auth MAUs), realtime concurrent connections (~200), Edge Function invocations
- Pro ($25/mo): Production ready: 8GB DB, 250GB bandwidth, daily backups, more Realtime, more compute.
- Team / Enterprise: SLAs, advanced security, dedicated infra.
- Add-ons: Compute upgrades, IPv4 add-on, log retention, branching minutes

Compute scales by upgrading the underlying Postgres instance size; storage scales independently.









