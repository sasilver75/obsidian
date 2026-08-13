AGE = "A Graph Extension"

A [[PostgreSQL|Postgres]] Graph database compatible with PostgreSQL's distributed assets and leverages graph data structures to analyze and use relationships and patterns in data.

Uses labeled vertices and labeled edges with property maps.
```SQL
CREATE
  (alice:Person {id: 1, name: 'Alice'}),
  (acme:Company {id: 50, name: 'Acme'}),
  (alice)-[:WORKS_AT {since: 2024}]->(acme)
```
- Each vertex has an internal graph identifier, a label, and an `agtype` property map.
- An edge has its own identifier, start and end vertex identifiers, a label, and `agtype` property map.
- AGE creates PostgreSQL schemas and relations for named graphs and labels.



Can be compelling when graph traversal is one part of a mostly relational system. It's not a drop-in, PostgreSQL-backed edition of Neo4j. 

```sql
-- Creat extension
CREATE EXTENSION age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Create a named graph
SELECT create_graph('social');

-- Invoke Cypher through a PostgreSQL set-returning function
-- AGE translates the Cypher query into PostgreSQL query-plan structures and executes it using PostgreSQL's transaction, storage, caching, and recovery mechanism. Graph values are returned using AGE's custom `agtype`, a superset/custom implementation of JSONB that can also represent vertices, edges, and paths.
SELECT *
FROM cypher('social', $$
    MATCH (p:Person)-[:WORKS_AT]->(c:Company)
    WHERE p.name = 'Alice'
    RETURN p, c
$$) AS (person agtype, company agtype);
```


If you have ordinary relational tables in Postgres:
```sql
CREATE TABLE accounts (
    account_id bigint PRIMARY KEY,
    region text NOT NULL,
    risk_score numeric NOT NULL
);

-- AGE can produce a graph traversal as a table-shaped result, which SQL can then join to accounts.
SELECT a.account_id, a.region, related.depth
FROM accounts AS a
JOIN (
    SELECT
        agtype_to_bigint(account_id) AS account_id,
        agtype_to_bigint(depth) AS depth
    FROM cypher('fraud', $$
        MATCH p = (:Account {id: 1001})-[*1..3]-(other:Account)
        RETURN other.id, length(p)
    $$) AS (account_id agtype, depth agtype)
) AS related
ON related.account_id = a.account_id
WHERE a.risk_score > 0.8;
```

# Migration from [[Neo4j]] concerns
- Neo4J uses Cypher (with extensive Neo4j-specific functionality), while AGE uses SQL containing `cypher(....)` calls, in AGE's implementation of openCypher (not identical).
- Property representation: Native graph property values in Neo4j vs `agtype` property maps inside AGE-managed PostgreSQL relations.
- Schema controls: Neo4j has and indexes, while AGE exposes PostgreSQL facilities plus AGE-specific graph/index behavior; not feature-for-feature equivalent.
- Graph tooling: Neo4j offers a mature browser, bloom, APOC, Graph Data Science, and managed offerings, while AGE offers an AGE Viewer... and has a markedly smaller ecosystem.
- Scaling and operations: Neo4j-specific clustering, routing, backup, and managed services. PostgreSQL operations, subject to AGE's compatibility with those facilities.
- Cypher Compatibility: 
	- Recent Neo4j syntax should be assumed incompatible until tested.
		- CALL { ... } subqueries
		- CALL ... YIELD
		- APOC procedures and functions
		- Graph Data Science procedures
		- Newer Cypher features or Neo4j-specific functions
		- Full-text and vector indexes
		- Schema constraints
		- Dynamic labels or relationship types
		- Shortest-path variants
		- Complex list and map expressions
		- Temporal, spatial, byte-array, and duration values
		- Very long or unbounded variable-length traversals
	- The `cypher()` result signature also means application code commonly needs to specify result columns and decode `agtype`.
	- In terms of ecosystem dependencies, APOC is often the hidden migration blocker. Inventory every APOC procedure and function before evaluating AGE.
- Performance capabilities:
	- Neo4j has the advantage for graph-dominant workloads involving deep, irregular, high-fan-out traversals.
	- PostgreSQL + AGE may have the advantage when filtering and aggregation are substantially relational, or when avoiding cross-database synchronization removes expensive work.
	- A graph stored within PostgreSQL still has storage and cache costs. One database does not mean "free graph operations."

# Migration Mechanics
- Inventory the Neo4J surface area:
	- Collect labels, relationship types, property types, constraints, indexes, every production Cypher query, APOC/GDS usage, drivers, transaction patterns, and operational requirements.
- Classify each query
	- Separate plain openCypher-like reads, writes, traversals, subqueries, procedures, etc.
- Create a representative proof of concept.
	- Import a production-shaped subset, including high-degree vertices and unusually large properties. A random 1% sample is often misleading because it may eliminate important hubs and paths.
- Create explicit durable IDs
	- Export nodes with stable business IDs. Export relationships using those IDs as endpoints. Do not depend on ephemeral database-internal identifeirs.
- Rebuild constraints and indexes deliberately.
	- Verify both correctness and whether `EXPLAIN` shows expected index usage.
- Port the application access layer
	- Replace Bolt/session APIs with PostgreSQL/AGE calls
- Run semantic differential tests
	- Execute logically-equivalent queries against Neo4j and AGE, normalize ordering and internal IDs, and compare complete results.
- Benchmark workload mixtures
- Dual run before cutover
- Retain a rollback path
	- A migration back out of AGE should be possible from durable vertex and edge exports.


# Migration Tooling
- There's an ecosystem, but there does not appear to be a mature, generally-accepted Neo4j -> AGE migrator. AGE's official migration surface is primarily its CSV bulk-loading functions plus language drivers.
- The surrounding tools can help build and operate a migration, but most do not understand enough Neo4j and AGE semantics to perform the migration automatically.


![[Pasted image 20260806133036.png]]




______________


An AGE graph contains:
- Vertices
- Directed Edges
- Properties on vertices and edges
- Paths, which are query results composed of alternating composed of alternating vertices and edges

A returned AGE vertex looks cocneptually like:
```
{
  id: 844424930131969,
  label: "ComponentType",
  properties: {
    name: "Radar",
    network: "CapeNet"
  }
}::vertex
```

An edge looks like:
```
{
  id: 1125899906842625,
  label: "CONNECTS_TO",
  start_id: 844424930131969,
  end_id: 844424930131970,
  properties: {
    confidence: 0.9
  }
}::edge
```

The official AGE model describes a vertex as having:
- A `graphid`
- One label name
- A property map

An edge has:
- `graphid`
- One label/type
- Two endpoint IDs
- A property map

When AGE creates a graph named `kg`, PostgreSQL gets a schema like:
```
kg
├── _ag_label_vertex   # Parent table
├── _ag_label_edge   # Parent table
├── ComponentType   # Each ordinary vertex or edge label gets its own PostgreSQL table
├── CapeNet  # Creating the label through Cypher can cause AGE to create the table automatically.
├── FormationInstance 
├── CONNECTS_TO
└── HAS_COMPONENT
```

So in standard AGE, `MATCH (n:ComponentType)` is conceptually asking PostgreSQL to look in the `ComponentType` vertex table.
By contrast, `MATCH(n)` means searching without a label, and may involve the parent table and its label-table descendants. ==This is why unlabeled property lookups are a potential full-graph scan concern.==

Properties are one `agtype` map
- AGE properties are not ordinary PostgreSQL columns by default. A label table conceptually contains something like:
```
id          graphid
properties  agtype
```
For example:
```
CREATE (:ComponentType {
    name: "Radar",
    network: "CapeNet",
    range_nm: 250,
    enabled: true
})
```
Stores the attributes together in an `agtype` property map:
```
{
  "name": "Radar",
  "network": "CapeNet",
  "range_nm": 250,
  "enabled": true
}
```
Where `agtype` is AGE's graph-value type. It's a superset/custom implementation related to [[JSONB]] and supports:
- Null, 64-bit integers, floating point numbers, exact numeric values, strings, lists, maps, vertices, edges, paths

So despite being PostgreSQL-based, AGE label tables are ==NOT== normally relational tables like:
```
ComponentType(
    id UUID,
    name TEXT,
    network TEXT,
    range_nm INTEGER
)
```
instead, they are closer to 
```
ComponentType(
    id GRAPHID,
    properties AGTYPE
)
```
==Properties therefore remain flexible and schema-less== (!!). Two ComponentType vertices can carry different property keys and values types unless we add validation ourselves.

So in terms of how to take multi-labeled Neo4j nodes and turn them into a single-label AGE node...there are several choices...
Given Neo4j:
```cypher
(:ComponentType:CapeNet {
    name: "Radar"
})
```
We could do:
1. One physical label, plus a classification property:
```cypher
(:ComponentType {
    kinds: ["CapeNet"],
    name: "Radar"
})

MATCH (n:ComponentType)
WHERE "CapeNet" IN n.kidns
RETURN n
```
In this case, the property filter needs an appropriate indexing strategy.

2. Most-specific physical label plus a supertype property
```cypher
(:CapeNet {
    entity_type: "ComponentType",
    name: "Radar"
})
```
This makes CapeNet the storage label but makes general "all component types" queries less natural.

3. Classification vertices and relationships
```cypher
(component:ComponentType)-[:MEMBER_OF]->(classification:Classification {
    name: "CapeNet"
})
```
This is graph native and extensive, but adds graph traversals and changes many queries.

4. Relational classification table
Because AGE is PostgreSQL, we could keep classifications relationally:
```
vertex_classification
---------------------
vertex_id
classification
```
Ten combine SQL and cypher.
This gives stronger relational constraints and indexes, but complicates pure-Cypher queries and application access.


### AGE Graph IDs
Note that AGE gives every vertex and edge a `graphid`, which is composed from:
- The entity's `label id`
- A sequence value belonging to that label.
These IDs are unique only within the graph; IDs can overlap between graphs.
This has important consequences:
- The physical label is encoded into identity
- IDs should not be treated as stable business identifiers.
- Moving or transforming label representation can change IDs
- IDs are unsuitable for long-live API references or migration joins...
So... ==we need properties such as the following, with separately enforced uniqueness==
```
{
    uuid: "c585985c-17c6-49b1-a176-e35372c2d937"
}
```


Thinking about Graph Validation:
1. Property shape: Does a `Platform` require `name`, `id`, and `platform_type`, with the data types for each property?
2. Identity: Is `Platform.id` present and unique?
3. Topology: Can a `CARRIES_SENSOR` only connect a `Platform` to a `Sensor`?
4. Domain Invariants: Must a scenario have exactly one root formation? Must timestamps be ordered? Is a cycle prohibited?...

Aside on databases with stronger schema enforcement:
- TypeDB: Probably the strongest native semantic validation; can reject invalid relationship participants and cardinality violations, but uses TypeQL and a different data model, so it would be a much larger product rewrite.
- TigerGraph: Much more rigid than Neo4j-style property bags, but omitted attributes can receive type defaults, so it doesn't mean that every meaningful field is required... and uses GSQL rather than Cypher.
- Stardog: RDF plus SHACL constraints; "guard mode" rejects transactions that violate shapes.
- ArangoDB: Good at required properties, types ranges, enums, and rejecting unknown fields... less expressive for graph-wide topology and semantic constraints.
- JanusGraph: Better than a totally open property bag, but not as semantically strict as TypeDB or SHACL guard-mode systems...


Ideas: "Hard Shell, Flexible Interior"
1. Create a versioned graph-schema manifest; make the schema authoritative in source control rather tahn discovering it from whatever data happens to be present. 
	- For each node type, we can define:
		- Physical AGE albel
		- Logical classifications
		- Required properties
		- Allowed properties
		- Property data types
		- Stable Identity Key
		- Uniqueness rules
		- Allowed relationship types
		- Ownership: Baseline, runtime, or derived (?)
		- Schema version
	- For relationship types:
		- Required properties
		- Allowed source node types
		- Allowed destination node types
		- Cardinality expectations
		- Whether duplicate edges are allowed
		- Stable identity rule, if applicable.
	- This manifest would then drive migrations, loaders, validators, documentation, and tests.
2. Promote critical fields into typed PostgreSQL structures
	- Don't leave identity and key operational fields buried only inside an arbitrary `agtype` map, instead.... add generated/extracted type columns and constraints around AGE label tables, or main tain separate relational idetntity/contract table keyed to teh graph object.
	- I don't like this idea!
3. Force writes throuhg a narrow gateway
	- A good schema is useless if arbitrary Cypher can bypass it. Instead, permit writes only through: Named Athena operations, Controlled migration/loader procedures, and later, a constrainer KG Curator adapter.
	- Avoid giving product roles a generic `RunCypher` write capability. One strong design is to expose PostgreSQL stored functinos or tightly-defined adapter methods that:
		- Validate the input
		- Update the relational registry
		- Execute the AGE mutation
		- Verify the mutation result
		- Commit everything atomically
		- ((Ehh I'm sort of wishy washy on these ones))
4. Use PostgreSQL constraints for row-local rules
	- UUID must exist and parse, Enum must contain an allowed value, Timstamp must have the correct type, valid_from <= valid_to, numeric value must be in range, required property must be present, property ap must not contain prohbited keys, identity must be unique within its logical type. PostgreSQL has some natiev check/unique/FK/constraint-trigger machinery.
	- Note that ==postgresql constraints CAN inspect values inside a JSONB docuement!== (though note that AGE uses `agtype` datattype, a sperset of JSONB, which )
5. Use triggers or procedures for topology rules.
	- Those normal CHECK constraints shouldn't query other rows. Rows such as these require a trigger, stored procedure, or relational contract table.
		- Things like (e..g., not real) `CARRIES_SENSOR` must connect a `Platform -> Sensor`, a graph must remain acyclic, a scenario must have exactly one root, etc... PostgreSQL triggers can inspect and reject writes, but complex traversal validation inside every write can become expensive. Perhaps enforce cheap topology rules synchronously and handle expensive global invariants during batch promotion or transaction-level validation.
6. Introduce the idea of staging and promotion, if it doesn't exist
	- Curator and bulk imports should not write directly into the serving graph.
	- A safer flow might be:
		- raw input -> typed staging tables -> schema and domain validation report -> quarantine invalid records -> transactionally promote data into AGE -> post-load checking of graph invariants (??)
7. Perhaps making unknown value explicit. 
	- A common source of KG trash is treating these as the same thing:
		- Property accidentally missing
		- Value genuinely unknown
		- Value not applicable
		- Value delierately redacted
		- Value not yet computed
	- For critical properties, perhaps we should encode status explicitly:
		- `range_value: null`
		- `range_value_status: "unknown"`
		- This makes both validation and downstream reading much clearer.
8. Run continuous graph-quality checks avnyways.
	- Write-time validation will not express everything cheaply. ==Maintain a graph-quality suite== for:
		- Orphan nodes and eddges
		- Invalid endpoint combinations
		- Missing required paths
		- Duplicate logical identities
		- Unexpected labels/properties
		- Mixed property types
		- Cycles where prohibited
		- Baseline/runtime ownership violations
		- Stale schema versions.
			- ==This is a good point==: If we have a central manifest, etc... what happens when we update it, such as adding a new required property? Does that mean that everything that's in the graph is invalid? Or do they stay valid because they were valid according to the schema at the time of their creation, or something?


# Aside on Postgres Check constraints: 

Assume we had anode table like (note that this is jsonb, not agtype)
```
CREATE TABLE platform (
    properties jsonb NOT NULL
);
```

Enforcing required fields
```sql
ALTER TABLE platform
ADD CONSTRAINT platform_required_properties
CHECK (
    properties ? 'id'
    AND properties ? 'name'
    AND properties ? 'platform_type'
);
```
Enforcing property types
```sql
ALTER TABLE platform
ADD CONSTRAINT platform_property_types
CHECK (
    jsonb_typeof(properties -> 'id') = 'string'
    AND jsonb_typeof(properties -> 'name') = 'string'
    AND jsonb_typeof(properties -> 'max_speed') = 'number'
);
```
Enforcing values and ranges:
```sql
ALTER TABLE platform
ADD CONSTRAINT platform_values_valid
CHECK (
    btrim(properties ->> 'name') <> ''
    AND (properties ->> 'max_speed')::numeric >= 0
    AND properties ->> 'platform_type'
        IN ('aircraft', 'ship', 'ground_vehicle')
);
```
Rejecting unrecognized properties:
```sql
ALTER TABLE platform
ADD CONSTRAINT platform_allowed_properties
CHECK (
    properties
        - ARRAY['id', 'name', 'platform_type', 'max_speed', 'source']
        = '{}'::jsonb
);
```
Enforce uniqueness with an expression index:
```sql
CREATE UNIQUE INDEX platform_id_unique
ON platform ((properties ->> 'id'));
```
The index prevents two documents from having the same `properties.id`.

One important subtlety: The `CHECK` passes when its result is either `TRUE` or `NULL`. Therefore, a type check by itself usually does NOT make a property required... you need an explicit existence check as well. 

## Aside on promoting critical JSON properties into Typed Columns
For important fields, generated columns can give a stronger and more convenient boundary:
```sql
CREATE TABLE platform (
    properties jsonb NOT NULL,

    platform_id uuid GENERATED ALWAYS AS (
        (properties ->> 'id')::uuid
    ) STORED,

    platform_name text GENERATED ALWAYS AS ( --Generated columns are values computed whenever the row changes
        properties ->> 'name'
    ) STORED,

    CONSTRAINT platform_id_required
        CHECK (platform_id IS NOT NULL),

    CONSTRAINT platform_name_required
        CHECK (btrim(platform_name) <> ''),

    CONSTRAINT platform_id_unique
        UNIQUE (platform_id)
);
```
This has useful behavior: A malformed UUID fails on insertion, a missing UUID fails the required check, duplicate UUIDs fail the unique constraint, the application can index and join using a real `uuid` column, the flexible property documents remain availabe.
- Generated typed columns can also participate in foreign keys:
```sql
source_platform_id uuid GENERATED ALWAYS AS (
    (properties ->> 'source_platform_id')::uuid
) STORED,

FOREIGN KEY (source_platform_id)
    REFERENCES platform(platform_id)
```

Q: These generated columns, are they like aliases or something? Waht does this let me do that I can't if I didn't "promote them"?
A: They're more than aliases; a generated column is araelt able column whose value Postgres computes from anotehr column whenever a row is inserted or updated. Promotion buys us:
1. A real PostgreSQL type
	- Inside JSON, the ID is just a JSON string, whereas the generated column is an actual `uuid`.
	- If someone then submits an `{"id": "garbage"}`, then the cast fails and the write is rejected.
	- This also applies to things like timestamps with time zones, integers, numeric, PostgreSQL enums, Arrays, and other domain-specific SQL types.
2. Ordinary constraints.
	- `CONSTRAINT platform_id_present CHECK (platofrm_id IS NOT NULL)`
	- `CONSTRAINT platform_id_unique UNIQUE (platform_id)`
	- The JSON remains flexible, but this particular property become required, typed, unique.
	- Q: Wait, isn't it the case that we could apply constraints to the JSONB properties anyways? What does this really functionally buy us by promotion to a generated column, in the case of constraints ? 
	- A: Yeah, you can enforce many constraints directly against JSONB properties without promotion... but what JSONB expressions generally cannot do is directly participate in column-oriented constraints. ((and then the singular example it basically gives me is foreign keys; other things like required, non-null status, type, range/enum, uniqueness, etc... are all doable against JSONB, it seems. So it ==SEEMS TO ME== like you'd basically just want to use this for foreign keys.))
3. Foreign Keys
	- This is one of the largest advantages; PostgreSQL foreign keys cannot normally reference arbitrary expressions like `properties ->> 'platform_id'`
	- After promotion, it's an ordinary column... so we can add `FOREIGN KEY (platform_id) REFERENCES platform(platform_id)` for instance.
4. Straightforward indexes and queries...
	- Without promotion, you have to do `WHERE (properties ->>'id')::uuid =$1`, while with promotion, it's a little simple `WHERE platform_id=$1`... and the indexing is similarly a little more conventional
		- Q: This seems cosmetic
5. One canonical extraction rule
	- Without a generated column, different queries may interpret the same value differently:
		- `properties ->> 'id'`
		- `(properties -> 'id')::text`
		- `(properties ->> 'id')::uuid`
	- ==Syntax NOTE:== `->` extracts a value while keeping it as JSON/`JSONB`, while`->>` extracts a value and converts it to SQL `text`.
		- So `(properties -> 'id')::text` first extracts the JSONB value `"abc-123"`, then serializes that JSONB as text, so the quote characters remain part of the result: `"abc-123"`. In contrast, `properties ->> 'id'` returns a simple `abc-123`.
		- As a result, this probably works: `(properties ->> 'id')::uuid`, while this probably fails: `(properties -> 'id')::text::uuid` because JSON tried to parse a UUID containing JSON quote `"` characters.
	- The generated column instead establishes one database-owned interpretation.

	- 

Note that you can do some of this without promotion; they're partly a convenience and clarity mechanism, not magic. 
- You can create an expression index directly, or use a JSON expression in a check:
```sql
CREATE UNIQUE INDEX platofrm_d_unique ON platform (((properties ->> 'id')::uuid));
CHECK ((properties ->> 'id')::uuid IS NOT NULL);
```
For simple validation and lookup, that might be sufficient, but genrated columns become especially valuable if you need foreign keys, repeated joins, several constraints on the same extracted value, straightforward application queries, one canonical conversion rule...

```sql
CREATE TABLE sensor_assignment (
    properties jsonb NOT NULL,

    platform_id uuid GENERATED ALWAYS AS (
        (properties ->> 'platform_id')::uuid
    ) STORED,

    FOREIGN KEY (platform_id)
        REFERENCES platform(platform_id)
);
```
Above: Example of benefit three, regarding foreign keys.


#### A side note on generated columns and their relationship to unknown values:
- Remember that there's a distinction:
	- Property accidentally missing
	- Value generally unknown
	- Value is not applicable (?)
	- Value deliberately redacted (?)
- A nullable generated value alone cannot distinguish those cases.. for important nullable properties, we would promote both the value and its status.


Relation to AGE:
AGE vertex-label tables conceptually contain something like:
```
id          graphid
properties  agtype
```
Because Cypher writes ultimately modify those Postgres tables, table checks and triggers should still see the writes... with AGE's property-access function, a ==required-property constraint== would conceptually resemble:
```sql
ALTER TABLE kg."Platform"
ADD CONSTRAINT platform_id_required
CHECK (
    ag_catalog.agtype_access_operator(
        properties,
        '"id"'::agtype
    ) IS NOT NULL
);
```
A ==typed validation== could extract the scalar, convert it to text, and the ncast it:
```sql
ALTER TABLE kg."Platform"
ADD CONSTRAINT platform_id_valid
CHECK (
    ag_catalog.agtype_to_text(
        ag_catalog.agtype_access_operator(
            properties,
            '"id"'::agtype
        )
    )::uuid IS NOT NULL
);
```
Here, a malformed UUID would cause the write to fail.
These examples are illustrative, rather than something I would paste into a production migration today. We need to test the exact expressions against the pinned AGE build...

Recommendations for AGE:
1. Checks directly on AGE label tables (for inexpensive row-local rules, e.g. required properties, valid UUID formats, nonempty strings, enum membership, etc.). These immediately reject malformed Cypher writes.
2. Relational registry for identities and relationships
3. Procedures/triggers for graph-wide rules that inspect multiple rows (`CARRIES_SENSOR` must connect `Platform -> Sensor`, etc.)


#### Re: #2, "Relational Registry for identities and relationships"
Conceptually AGE, creates tables like:
```
Platform
├── id          graphid
└── properties  agtype

Sensor
├── id          graphid
└── properties  agtype

CARRIES_SENSOR
├── id          graphid
├── start_id    graphid
├── end_id      graphid
└── properties  agtype
```
When you run
```cypher
MATCH (p:Platform), (s:Sensor)
WHERE p.uuid = $platform_id
  AND s.uuid = $sensor_id
CREATE (p)-[:CARRIES_SENSOR]->(s)
```
AGE creates an edge whose start_id and end_id reference the internal AGE graph IDs of those vertices... Traversal use those internal references efficiently:
```cypher
MATCH (p:Platform)-[:CARRIES_SENSOR]->(s:Sensor)
RETURN p, s
```
So... while AGE knows the edge's internal source/destination vertex, the relationship label, and its properties, it doesn't inherently know your domain rules, such as "`Carries_Sensor `may only connect a platform to a sensor."... These would be the gaps that a registry would intend to fill.

Q: Wait, is the idea that basically #1 is these single-node constraints, for instance... and that #2 and #3 kidn of work together for these multi-node constraints?


#### Aside: How do Postgres triggers work, again?
- Postgres triggers are synchronous by default and can absolutely block or reject writes.
- A stored proecure/function starts when the application explicitly calls it.
- A `BEFORE` trigger is fired by the database automatically, before each row is written (well before commit, in the transaction). Typically used to validate/normalize/change/skip a row.
- An `AFTER` trigger is fired after the row/statement, but before commit. Typically used to validate final values ,update related tables, audit.

Stored Procedure: Like a datbase-side service method:
```sql
CALL kg_create_relationship(
    relationship_kind := 'CARRIES_SENSOR',
    source_id := '...',
    target_id := '...'
);
```
This might confirm that source exits, confirms that target exists, confirm that source is a platform, that target is a platform, confirm that the relationship is allowed, insert the relationship registry row, create the AGE edge, and return success. If step 6 fails, the transaction rolls back. The important part to know though is that ==calling a stored procedure is voluntary;== if a database roe is allowed to issue arbitrary Cypher or directly update the tables, it can bypass the procedure.

Trigger: Automatically attached to a table event; every matching table write fires it, regardless of whether the write came from Athena, a loader, a procedure, direct SQL, etc.
```sql
CREATE TRIGGER validate_relationship
BEFORE INSERT OR UPDATE
ON kg_relationship
FOR EACH ROW
EXECUTE FUNCTION validate_relationship();
```
The trigger function can inspect `NEW` and `OLD`, query other tables, modify the proposed row, or raise an exception:
```sql
CREATE FUNCTION validate_relationship()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.source_entity_id = NEW.target_entity_id THEN
        RAISE EXCEPTION
            'Self-relationships are not permitted for %',
            NEW.relationship_kind;
    END IF;

    RETURN NEW;
END;
$$;
```
An invalid write fails immediately, and nothing is committed:
```sql
ERROR: Self-relationships are not permitted for CARRIES_SENSOR
```
A row-level ==`BEFORE` trigger== runs immediately before each row is inserted, updated, or deleted.
An ==`AFTER` trigger== runs after the row has been changed, but still within the transaction. At that point, the new row exists, other changes made by the statement are visible, etc. 
Postgres also supports an ==`AFTER STATEMENT`== trigger that can use transition tables to inspect the whole set of rows changed by that statement, which can be better for bulk loading...
Postgres has deferred constraint triggers. A deferred constraint trigger handles invariants that cannot be evaluated correctly until all related changes in the transaction are complete...

Note that using triggers for expensive traversal validation (e.g. detection of cycles) for every inserted edge can be very expensive, increasing write latency. Poorly designed triggers can even recursively invoke themselves, cause unexpected writes, make failures difficult to trace, perform badly during imports, break extension upgrades, etc. They should be named, tested, and documented.





[Video: Bringing GRahp to PostgreSQL: A Deep Dive into Apache AGE](https://www.youtube.com/watch?v=2WC2dG2pmHc)
See also the blog series: [Part 1](https://gdotv.com/blog/apache-age-explained/), [Part 2](https://gdotv.com/blog/running-apache-age-docker-cloud/), [Part 3](https://gdotv.com/blog/loading-data-apache-age-air-routes/), [Part 4](https://gdotv.com/blog/visualizing-apache-age-gdotv/)
> "Allows you to run a mix of graph workloads and regular SQL workloads"
> "Allows you to create Graph data directly on PostgreSQL. Imagine that you have a dataset that's not well suited to... deep traversals; say, anywhere from 3-6 hops, and you're trying to get something set up that matches your existing infra.... with Postgres being one of the most popular databases, there's a good chance that you either have it in production or understand how to get it into production. Create graphs, load data into them, and get querying using Cypher."
> "When you create nodes and edges, they're created under a specific graph declared beforehand, and that graph has a schema in your postgres instance that's used to store tables that store the actual nodes and edges data. It simply stores one row per node and edge in separate tables for each label for your database, and stores properties as JSONB values within those rows, allowing you schemaless flexibility, etc."
> "For large scale graph workloads.... millions or billions of nodes... AGE is not the right choice; it's not an optimized-to-the-max solution when it comes to graph workloads; instead, you should look at it as a convenient option for smaller graphs that you want to deploy rapidly on existing infrastructure... For those, you should look at Neo4j, FalcorDB, Memgraph."
> "SQL PGQ is a property graph querying feature set that's part of the SQL 2023 standard that allows you to declare property graphs on top of existing relational tables, and to then query it via Graph queries. When it comes to SQL PGQ... it's not available at time of recording (June 2026) on postgres just yet; will be released at the ned of 2026 as part of Postgres19; PGQ uses standard SQL language for graph queries with some new functionalities with the language that let you do that... And it works on top of your pre-existing relational data, so what that means is that you already have tables on your postgres instance and you're trying to deploy your graph workload on top of those tables... You just have to declare a property graph on top of your existing data. SQL PGQ on Postgres ha some similar limitations around performance, so comparisons with other graph databases... still apply. It has an added advantage over Apache AGE which is that it layers over pre-existing data."
> So how do we run AGE?
> It's a Postgres extension, but not an *official postgres extension.* There are some caveats: The main caveat that come with it beign a ==community extension== is that depending on where you're deploying postgres in production, you might not have access to it. If you're managing your own Postgres instance in your own environment, you're good to go.
> One of the ... pioneers in supporting this specific project in Microsoft Azure... who landed official support of AGE in 2025 in their Azuer Database for Postgres, but also for Azure HorizonDB, which is in early access preview mode.
> "Loading nodes in Apache AGE is very simple by CSV; where they're mores subtlety involved is when we want to laod edges, which connect nodes. To create an endge, we need to know our start node, where we're connecting from, and our end node, where we're connecting to. The Apahce AGE csv loader has a requirement that the format that you need the identifier of the start and the end node, but you also need to know what vertex type that you're connecting fro/to is well."



# Comparison with PostGraphile, Hasura
- Postgraphile and Hasura offer similar propositions, where they convert PostgreSQL schemas into GraphQL APIs... lacks the built-in capability for complex graph data handling.

