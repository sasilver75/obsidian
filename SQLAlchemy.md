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

