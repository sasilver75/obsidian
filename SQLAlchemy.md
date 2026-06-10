
A Python database toolkit, giving two related things:
1. SQLAlchemy Core: A Pythonic SQL construction and execution layer
2. SQLAlchemy ORM: An object layer that maps Python classes to database tables. Sits on top of Core.

The modern style is the "2.0 style" using `select()`, `Session.execute()`, typed declarative models, and explicit transaction boundaries.

Think of SQLAlchemy as a stack:
```
Your app code
  -> SQLAlchemy ORM Session, mapped classes, unit of work
  -> SQLAlchemy Core: select(), insert(), Table, Column, Connection
  -> Engine
  -> Pool
  -> Dialect
  -> DBAPI driver
  -> Database
```
Above: [[Unit of Work]], [[DBAPI]]

### Engine
The ==Engine== is the central database access object. You usually create one per database URL, near application startup time:
```python
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://user:pass@localhost/app)
```
Above: `postgresl` is the database backend, `psycopg` is the [[DBAPI]] driver, the the rest is location/auth

This Engine is not usually a "connection," it more like a database access factory; it knows how to create connections, owns a connection pool, knows the database dialect, and provides the main entry point for executing SQL through Core or supplying connections to ORM sessions.

### Dialect
- The ==dialect== is SQLAlchemy's adapter for a specific database family and driver combinations. 
- The various popular SQL databases all have slightly different SQL, parameter styles, feature support, etc. SQLAlchemy uses dialects to answer: "How do I render this Python SQL expression into a SQL expression that database X understands?"

### [[DBAPI]] /  Driver
- The DBAPI driver is the lower-level Python package that actually talks to the database
```
psycopg / psycopg2 -> PostgreSQL
sqlite3 -> pysqlite -> SQLite
mysqlclient / pymysql -> MySQL
asyncpg -> async PostgreSQL
```
SQLAlchemy does not itself open raw [[Transport Control Protocol|TCP]] sockets to Postgres during ordinary use; it delegates that to a driver.

#### Pool
- The ==Pool== manages physical database connectiosn for reuse; actually opening database connections is expensive/slow, so SQLAlchemy checks them out from a pool and returns them when done.
- Most synchronous engines use `QueuePool` by default for typical server databases. Async engines use an async-adapted queue pool.
```python
create_engine(
	url,
	pool_size=5,
	max_overflow=10,
	pool_pre-ping=True, # Checks whether a pooled connection is alive before using it
	pool_recylce=1800,
)
```
When you call `connection.close()` on a pooled SQLAlchemy connection ((or rather, when your session does it)), that often actually means "return this to the pool."

### Connection
- A ==Connection== is a SQLAlchemy Core's handle to an actual checked-out DB connection
```python
from sqlalchemy import text

with engine.connect() as conn:
	result = conn.execute(text("select 1"))
```
You can use a `Connection` directly when writing Core-style SQL or lower-level database code.

### Transaction
A transaction is the database's unit of [[Atomicity|Atomic]] work -- all changes commit, or they roll back.
```python
with engine.begin() as conn:
	conn.execute(...)
	conn.execute(...)
```
Using `engine.begin()` gives you a connection (like engine.connect) but *also* wraps that block in a transaction! If the block succeeds, it commits. If an exception occurs, it rolls back.


### SQLAlchemy Core
SQLAlchemy's SQL expression system; instead of concatenating SQL strings, you build SQL with Python objects:
```python
from sqlalchemy import select

stmt: Select = select(User).where(User.name == "alice")

# Or you can do it for tables
stmt: Select = select(user_table).where(user_table.c.name == "alice")

# You can pass these Select objects to a Conn.execute() or Session.execute()
# SQLAlchemy then compiles it into real SQL for your databae dialect.
```

### MetaData, Table, Column
```python
from sqlalchemy import MetaData, Table, Column, Integer, String

# A collection of table definitions
metadata = MetaData()

# Represents a database table
user_table = Table(
	"user_account",
	metadata,
	Column("id", Integer, primary_keys=True), # Represents a database column
	Column("name", String)
)
```


### ORM
The ORM maps Python classes to database tables! Instead of manually handling rows, you worok with objects:
```python

user = User(name="alice")
session.add(user)
session.commit()
```
The ORM is not magic storage. It is a mapping layer over SQL. It still emits INSERT, SELECT, UPDATE, and DELETE.

### DeclarativeBase
- A modern Base Class for defining ORM models

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
	pass
	
```
Every mapped model subclasses `Base`.

#### Mapped and mapped_column
- SQLAlchemy 2.0 leans into Python typing!

```python
from sqlalchemy.orm import Mapped, mapped_column

class User(Base):
	__tablename__ = "user_account"
	
	# Mapped says "This ORM attribute is mapped by SQLA and has Python type int!"
	# mapped_column configures the actual database column
	id: Mapped[int] = mapped_column(primary_key=True) 
	name: Mapped[str]
	
```

### Mapper
- The `Mapper` is the internal object that connects a Python class to a database table or selectable.
	- You usually don't create these directly in modern declarative code.
When you write
```python
class User(Base):
	__tablename__ = "user_account"
```
SQLAlchemy configures a mapper behind the scenes.

The mapper knows things like: this class uses this table, this column maps to this attribute, this relationship connects to that class.

### Relationships
We can describe links between mapped classes

```python
class User(Base):
    addresses: Mapped[list["Address"]] = relationship(back_populates="user")

class Address(Base):
    # The actual FK that is a database constraint
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
    # The ORM-level object navigation and persistence behavior
    user: Mapped["User"] = relationship(back_populates="addresses")
```
Relationships are connecting units between mapped classes that handle loading references/collections and persisting linkages.
- They're ORM-level attributes that connect one mapped Python object to another mapped Python object. It's not a database column; we typically have a separate FK column though to the table with we have a relationship (on at least one side of the relationship, at least)

```python
class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)

	# back_populates tells SQA: "This relationship and that other erlationship are two sides of the same link!"
    addresses: Mapped[list["Address"]] = relationship(
        back_populates="user"
    )


class Address(Base):
    __tablename__ = "address"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

	# back_populates tells SQA: "This relationship and that other erlationship are two sides of the same link!"
    user: Mapped["User"] = relationship(
        back_populates="addresses"
    )


# This lets us do:
address.user # the user for this address
user.addresses # the list of Address objects for this user
# Be careful of how you load these! We don't want to do an N+1 problem for mapping through user.addresses!
```
- Relationships can be many-to-one, one-to-many, one-to-one, or many-to-many


### Session
- The `Session` is the main ORM working object. It is where ORM queries happen, where new objects are added, where changed objects are tracked, and a place through which transactions are committed or rolled back.

```python
from sqlalchemy.orm import Session

with Session(engine) as session:
	user = User(name="alice")
	session.add(user)
	session.commit()
```
A good short definition: ==The `Session` is a transaction-scoped workspace for ORM objects==

The official docs call it a "holding zone" for objects loaded or associated with it. It gets a connection from the `Engine` when needed, opens a transaction, flushes changes, commits or rolls back, and then releases the connection back to the pool.

#### sessionmaker
A factory for sessions!
```python
from sqlalchemy.orm import sessionmaker

# We call the returned object SessionLocal by convention, because it's used to make local (to a request, script, job) Session!
SessionLocal = sessionmaker()

# This context manager setup 
with SessionLocal() as session:
	... # Do stuff with the session!
```

In applications, you usually create the engine once, create a session factory once, and then create one `Session` per request/job/unit of work.

### Identity Map:
- A dictionary-like internal structure inside the `Session`. 
	- It ensures that within one session, one database row corresponds to one Python object instance.
	- If you load `User(id=1)` twice in the same session, SQLAlchemy tries to give you the same Python object, not two separate objects representing the same row (weird split brain local effects)
- This matters because ==object mutation then has a coherent meaning!==

```python
u1 = session.get(User, 1)
u2 = session.get(User, 1)

u1 is u2  # usually True within same Session
```


### [[Unit of Work]] Pattern
- One of the most important ORM ideas!
You don't say:
```SQL
UPDATE user SET name = ...
```
Instead, we mutate objects!
```python
user.name = "Alice Smith"
```
The `Session` object then records that change.
Later on, `flush()` or `commit()`, SQLAlchemy calculates the SQL needed to synchronize the database with the changed object graph!

So Unit of Work is:
- Load/crate objects
- Mutate objects
- Session tracks changes
- Flush emits SQL in dependency order
- Commit finalizes transaction

The docs state that ORM objects are instrumented so that attribute changes produce change events recorded by the `Session`; before querying or committing, the session flushes pending changes.


### Flushing vs Committing
- `flush()` sends pending SQL to the database inside the current transaction
- `commit()` flushes first, then commits the transaction
```python

print(user.id) # None (it hasn't been persisted)
session.add(user)

session.flush() # INSERT happens; transaction still open, though!
print(user.id) # 34s1f3...  (DB-generated fields are populated!)

session.commit() # Transaction committed
```
After flush, (e.g.) generated primary keys are available, but other transactions shouldn't treat the data as durable until commit (depends on your Database, isolation settings, etc)

==IMPORTANT:==
- Flush always writes whatever pending INSERT/UPDATE/DELETE SQL is needed for dirty objects in the session.
- SQA must know generated PK after INSERT, so after flush ==.user_id is normally populated==, but something like ==.created_at may not be==. That might require a refresh.
A common pattern you'll see is documentation is:
```
user = User(...)
session.add(user)
session.commit()
session.refresh(user)
```
But only do this if you need to; if you only need `id`, maybe not!

`flush()`: "Send my pending SQL, then fill in whatever SQAlchemy either already knows or can cheaply/necessarily get back."

```python
class User(Base):
	id = mapped_column(primary_key=True)
	email = mapped_column(String)
	
	# Python-side default: SQLAlchemy runs this. Set at flush time, not obj creation 
	# Will be populated post flush(), because SQA generates it before sending INSERT
	created_at_py = mapped_column(DateTime, default=datetime.utcnow)
	
	# Server-side default: Database server runs this
	# MAYBE populated after flush, depends if SQA fetched it with RETURNING
	# Not guaranteed on every backend/config
	created_at_db = mapped_column
```
Populated by flush reliably:
- Explicitly assigned values
- Generated primary keys
- Python-side defaults: default=..., onupdate=...
- Relationship foreign keys once generated parent IDs are known



### Autoflush
- Means that the session may *automatically flush pending changes* before a query, so that the query sees the database state that corresponds to your in-memory changes.
	- ((It seems to me that there are certain actions that might require/benefit from your dirty-but-not-yet-flushed objects being flushed, so these certain actions trigger a flushing))
==This can be pretty surprising!== If you created an invalid object and then issue a SELECT, SQLAlchemy might try to INSERT your invalid object first!
You can temporarily suppress it with:
```
with session.no_autofllush:
	...
```

### Commit, Rollback, and Close
Methods on `Sesssion` objects
- `commit()`: persists the transaction
- `rollback()`: undoes the current transaction and clears the failed transactional state
- `close()`: releases connection resources and detaches/cleans up session's current state.

Typical write pattern:
```python
with Session(engien) as session:
	try:
		session.add(obj)
		session.commit()
	except:
		session.rollback()
		raise
```

### Object States
ORM objects move through states:
```
transient: plain Python objects, not yet in a session
pending: Added to session, not flushed yet
persistent: Associated with session and database identity
deleted: Marked for deletion
detached: No longer associatd with a session
expired: attributes cleared; will reload on access
```
"Detached Object" errors usually happen when you try to lazy-load something after the session is gone

==Lazy loading== means SQLAlchemy loads related data when you first access it:
```
user.addresses # may emit SELECT here
```
==Eager loading== means you ask SQLAlchemy to load related data up front:
```
from sqlalchemy.orm import selectinload
stmt = select(User).options(selectinload(User.addresses))
```

Common eager strategies:
- `selectinload`: Usually a great default for collections
- `joinload`: Joins related rows into the original query
- `subqueryload`: Older/special-case strategy

Be intentional! ==Lazy loading can cause [[N + 1 Query Problem]]==!

### Result, Row, ScalarResult
- When you execute a statement, you get a `Result`
```
result: Result = session.execute(select(User))
```
Rows by default:
```
rows = result.all()
```

If you selected one ORM entity and want the entity objects directly, from the response:
```python
users: list[User] = session.scalars(select(User)).all()
```

Common methods on `Result` objects:
```python
result = session.execute(select(User))

result.all() # list[Row[tuple[User]]]
result.first() # Row[tuple[User]] | None
result.one() # Row[tuple[User]] or raises NoResultFound/MultipleResultsFound
result.one_or_none() # Row[tuple[User]] | None or raises MultipleResultsFound
result.scalar() # User | None ... first "column" of first row ((User,))
result.scalar_one_or_none() # User | None or raises MultipleResultsFound


# If instead we used session.scalars, we don't have this (User,) Row tuple wrapping
result = session.scalars(select(User))

result.all() # list[User]
result.first() # User | None
result.one() # User or raise NoResult/MultipleResultsFound
result.one_or_none() # User | None or raise MultipleResultsFound
```


### Core queries vs ORM Queries

```python

# In core, we can refer to the user_table object and drill to the column name
stmt = select(user_table).where(user_table.c.name == "alice")

with engine.connect() as conn:
	rows: list[Row] = conn.execute(stmt).all()
	

# In the ORM
stmt = select(User).where(User.name == "alice")

with Session(engine) as session:
	users: list[User] = session.scalars(stmt).all()
```

### Async SQLAlchemy

Async uses async versions of these same ideas!
```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/app") # New!
SessionLocal = async_sessionmaker(bind=engine) # New!

session: AsyncSession = SessionLocal()

async with SessionLocal() as session:
	result = await session.scalars(select(User))
	users = result.all()
```
Note that AsyncSession is still a session. Treat it as one unit-of-work object, not as a global shared object across concurrent tasks.


### Alembic
[[Alembic]] isn't SQLAlchemy itself, but it's the standard migration tool for the same ecosystem
- SQLAlchemy models describe a database schema, and Alembic manages schema changes over time.

You can use SQLAlchemy to define and use tables, and use Alembic to over time add columns, create indexes, rename tables, migrate production safely.


### How SQLAlchemy is usually used
```python
# Somewhere, creating a sync SQA engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Perhaps elsewhere, defining models
class User(Base):
	__tablename__ = "user_account"
	
	id: Mapped[str] = mapped_column(primary_key=True)
	name: Mapped[str]

# Elsewhere, run a create a user
# In this example, they're creating the session very locally to the query; it's typically created in a route handler I think for FastAPI
def create_user(name: str) -> User:
	with SessionLocal() as session:
		user = User(name=name)
		session.add(user) # Adds to session, but doesn't flush
		session.commit() # Flushes and commits; user now has an .id
		session.refresh(user)
		
# Elsewhere, run a query
# Again, this session is typicaly per-request in FastAPI, not handled here
def get_user(user_id: int) -> User | None:
	with SessionLocal() as session:
		return session.get(User, user_id)

# More complex query
stmt = select(User).where(User.name.ilke("%alice%")).order_by(User.id)
with SessionLocal() as session:
	users = session.scalars(stmt).all()
```

### The Big Picture

If you remember only one flow:
- ==Engine== owns the ==Pool== and ==Dialect==
- ==Pool== manages the reusable DBAPI connections
- ==Connection== executes Core SQL in a transaction
- Declarative models map Python classes to tables
- ==Session== manages ORM objects, acquires connections, and manages transaction
- ==Identity map== keeps one object per row per session
- ==Unit of Work== is the pattern that tracks changes and flushes SQL
- Commit makes the transaction durable, or Rollback abandons it


____________

A Python library with ==two distinct layers== (you can use either or both):
- ==Core== (low-level)
	- A ==SQL expression builder==. Lets you construct and execute SQL programmatically with Python objets instead of raw strings.
	- Handles [[Connection Pool|Connection Pooling]], parameter binding (preventing [[SQL Injection]]), and dialect differences between databases (e.g. [[PostgreSQL|Postgres]] vs [[MySQL]]).
```python
from sqlalchemy import select, text

# Build a query as a Python expression
stmt = select(SR311).where(SR311.h3_r8 == "882a100d2dfffff")

# Or drop to raw SQL when needed
result = await session.execute(text("SELECT COUNT(*) FROM public.sr_311"))
```
- ==ORM== (high-level)
	- ==Maps Python classes to database tables==. You define a ==Model== class, and SQLAlchemy handles generating the SQL for inserts, updates, selects, and deletes.
		- Every model inherits from a shared `Base` class which carries a `metadata` object that accumulates a description of every table defined in every subclass.
	- We use the ORM for defining schema, which tools like [[Alembic]] reads, and for simple queries. 
		- Alembic reads a model's `Base.metadata` to know what the schema should look like.
```python
class SR311(Base):
    __tablename__ = "sr_311"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sr_number: Mapped[str] = mapped_column(Text, unique=True)
    h3_r8: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
```


### Mapped Columns
- In SQLAlchemy 2.0 (released 2023), we got a new style for defining models using Python type annotations:
```python
from typing import Optional
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Text, BigInteger, Boolean

class SR311(Base):
    __tablename__ = "sr_311"

    # Mapped[T] declares the Python type of the attribute
    # mapped_column() declares the SQL column properties
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    sr_number: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    h3_r8: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    has_valid_coords: Mapped[bool] = mapped_column(Boolean, default=False)
```
- Above: `Mapped[str]` means that the Python attribute is a str, for instance. SQLAlchemy can infer `nullable` on a column from an `Optional` python type annotation.



# Core Lifecycle

### Objects
- ==Engine==
	- Owns the [[Connection Pool]]
	- Produces DB connections
- ==sessionmaker== / SessionLocal
	- A factory for Sessions
- ==Session==
	- A request/use-case scoped ORM workspace
	- ***Borrows a connection when needed***
	- Tracks objects and pending changes
	- Flushes SQL
	- commits/rolls back transaction
	- closes and returns connection to pool

```python
session.add(obj)
sess.execute(stmt)
session.flush()
session.commit()
session.rollback()
session.close()
```
Important: A `Session` is not just a connection, it's a workspace that tracks ORM objects and pending changes.





### sessionmaker and SessionLocal
```python
# We call it SessionLocal by CONVENTION because it creates sessions local to a request, job, script, etc. It is not itself a session, it creates sessions.
SessionLocal = sessionmaker(bind=engine)

# Create the session
session = SessionLocal()

# Typically, in (eg) FastAPI, we do this via a context manager + Depends that injects it into our wrote, which handles the session creation and closing.
def get_session():
	# This context manager uses the SessionLocal's __enter__/__exit__ methods to clean up
	with SessionLocal() as session:
		yield session

# And then in our route-handling function...
def handle_route(session: Session = Depends(get_session))
	...

# Now the context manager calls session.close() automatically after the request finishes.
```





Flush: To "flush" menas:
> Send pending ORM changes from the SQLAlchemy Session to the database as SQL, without committing the transaction.

```python
truck = FoodTruck(name="Al Pastor Atlas")
session.add(truck)
# At this point, SQLAlchemy is trackign truck, but it may not have emitted SQL

session.flush()
# SQLAlchemy sends "INSERT INTO food_trucks ..." but the transaction is still open
# Might raise a unique constraint error, not-null error, etc.
print(truck.id) # PK will be present
print(truck.created_at) # May not be present 

# Forces a SELECT to reload the object from the database
session.refresh(truck)
print(truck.createdD_at) # Present

# session.rollback()
# You could roll back the session if you'd like.

session.commit() # Flushes automatically first, but also commits


# Later: The Session is done using its check-out connection, and can return it to the pool.
# In FastAPI, the common pattern is a dependency/context manager that handles this and calls session,.close() automatically after the request finishes.
session.close()
```
==IMPORTANT:==
- Flush always writes whatever pending INSERT/UPDATE/DELETE SQL is needed for dirty objects in the session.
- SQA must know generated PK after INSERT, so after flush ==.user_id is normally populated==, but something like ==.created_at may not be==. That might require a refresh.
A common pattern you'll see is documentation is:
```
user = User(...)
session.add(user)
session.commit()
session.refresh(user)
```
But only do this if you need to; if you only need `id`, maybe not!
More on this in the flush section, I'll copy this.


Lifecycle:
- Engine owns the connection pool
- Session is created
- Session does not necessarily have a DB connection yet
- Session does its first query/flush
- Session checks out a connection from the pool
- Session uses that connection
- Session commit/rollback ends transaction
- Session.close() releases connection back to the pool
- Connection remains open in pool for reuse
