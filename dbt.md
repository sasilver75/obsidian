---
aliases:
  - Data Build Tool
---
An open-source transformation framework that allows data analysts and engineers to write `SQL SELECT` statements and have `dbt` handle the rest: 
- Materializing these queries as tables or views in your [[Data Warehouse]]
- Managing dependencies between transformations
- Testing data quality
- Generating documentation

It's become ==the dominant tool for the transformation layer in modern daata stacks== (think the T in [[ETL|ELT]]).

DBT's philosophy is simple: 
> **"transformations should be defined as SELECT statements, and the tool handles the DDL/DML for you.***

The fundamental unit in dbt is a ==model==, a SQL file containing a `SELECT` statement. Each model corresponds to a table in your warehouse.

==Materialization types== determine how `dbt` builds the model in the warehouse:
1. `view`: Creates a SQL view. No data stored, query runs on access. Good for lightweight transformations on small data.
2. `table`: Drops and recreates the full table on every run. Simple, but expensive for large tables.
3. `incremental`: On first run, build the full table. On subsequent runs, only process new/changed records and appends/merges them. *Essential* for large tables.
4. `ephemeral`: Not materialized at all; injected as a CTE into models that reference it. Good for intermediate transformations you don't need to store.
5. `materializedview`: Newer, uses the warehouse's native materialized view features where supported.


The ==ref()== function is how models reference eachother
- DBT parses all ref() calls across all models and builds a [[Directed Acyclic Graph]] (DAG) of dependencies:
```
raw_events                                                                        
    └── bronze_events                                                             
          └── silver_sessions
                ├── gold_conversion_rate                                          
                └── gold_session_metrics                                          
                      └── gold_executive_dashboard
```
`dbt` executes models in the correct order automatically; if `gold_conversion_rate` depends on `silver_session`, dbt *always* builds `silver_session` first, and parallelizes what can be parallelized. and serializes what must be serialized.

DBT models typically aren't plain SQL, they're [[Jinja]] templates, which adds programmatic capabilities.

==Macros== are reusable Jinja functions, the equivalent to stored procedures or functions but in `dbt`'s templating scheme. There's a rich ecosystem of macros in `dbt-utils`` and other packages.

DBT has a ==built-in testing framework== for data quality:
- Generic tests: Applied to columns via [[Yet Another Markup Langauge|YAML]] configurations
- Singular tests: Custom SQL tests; if it returns rows, it fails.

`dbt` ==autogenerates a documentation site== from your models, YMAL descriptions, and lineage graph.


# DBT on the Modern Data Stack:
```
  Sources (databases, SaaS APIs)
    → Airbyte/Fivetran (ingestion → raw tables in warehouse)                      
      → dbt (transformation → bronze/silver/gold tables)                          
        → BI tools (Looker, Tableau, Metabase)                                    
        → ML pipelines (feature engineering)                                      
        → Reverse ETL (Hightouch, Census → CRM, ad platforms)                     
```



![[Pasted image 20260425213714.png]]