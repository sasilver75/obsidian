
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
	- 







