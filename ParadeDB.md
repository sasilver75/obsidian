A Postgres-based search and analytics platform, built as a set of Postgres extensions plus a packaged distribtion (Docker image, managed cloud) so that you keep your data in Postgres but get full-text search, faceted seach, and analytical queries.

[[pg_search]] ==is ParadeDB's flagship extension! It's hte part that delivers full-teext and hybrid search.==
- ==ParadeDB is the company/product. pg_search is their open-source Postgres extension that does the search work.== 
	- Other extensions in their stack include:
		- `pg_analytics`: Columnar/OLAP via [[DuckDB]]
		- `pg_lakehouse`: query [[Apache Parquet|Parquet]]/[[Apache Iceberg|Iceberg]] from Postgres

## Why it matters
- The standard pattern for "Postgres + Search" is to *dual-write* to [[ElasticSearch]], deal with sync lag, schema drift, and operational overhead. ParadeDB lets you keep one system, get Elastic-quality search, drop the dual-write.

> *"Simple, Elastic-Quality Search for Postgres You want better search, not the burden of Elasticsearch. ParadeDB is the modern Elastic alternative built as a Postgres extension."*


> The Complete Toolkit for Search

### Text Processing
```sql
CREATE INDEX ON animals
USING bm25 (
    id,
    (name::pdb.ngram(3,3)),
    (description::pdb.unicode_words('stemmer=english'))
);
```
- 12+ tokenizers to break apart text, support for 20+ languages, including dictionary-based tokenizers

### Text Search
```sql
SELECT * FROM animals
WHERE name &&& 'asian elephant'
OR id @@@ pdb.more_like_this(1)
LIMIT 5;
```
- Full text search (match, phrase, term, and fuzzy queries), and many more like proximity, more-like-this, regex, etc

### Hybrid Search
```sql
SELECT id, pdb.score(id)
FROM animals
WHERE name &&& 'asian elephant'
ORDER BY pdb.score(id) DESC
LIMIT 5;

SELECT id, embedding <=> '[1,2,3]'
FROM animals
ORDER BY embedding <=> '[1,2,3]' DESC
LIMIT 5;
```
- Can be combined with [[pgvector]] to deliver a hybrid search solution.

### Boolean Queries
```sql
SELECT * FROM animals
WHERE name &&& 'asian elephant'
AND metadata->>'region' === 'Asia'
AND weight >= 4000
LIMIT 5;
```
- Multiple boolean conditions handled asa a single index scan, metadata pre-filtering of results

### Top K
```sql
SELECT * FROM animals
WHERE name &&& 'asian elephant'
ORDER BY weight DESC
LIMIT 5;
```
- Efficient Top K, tunable relevance scores with [[BM25]]

### Aggregates
```sql
SELECT metadata->>'region', count(*)
FROM animals
WHERE name &&& 'asian elephant'
GROUP BY metadata->>'region'
ORDER BY 1;
```
- Bucket and metric aggregates (count, bucket, average, ...) with a columnar index.
- Return aggregate alongside search results in a single query

