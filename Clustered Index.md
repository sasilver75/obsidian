An index whose leaf level is the table data itself, rather than pointers to the table data on disk. Its ordering determines the physical storage ordering of the table on disk.


# Comparison  with [[Primary Index]]
- A Primary Index is just an index on the [[Primary Key]] of a table.
- A clustered index determines the ordering in which the table data is laid out on disk.
They're separate axes! They can overlap, but they don't have to.

In MySQL's InnoDB storage engine, the `PRIMARY KEY` index is clustered by default
In SQL Server, a `PRIMARY KEY` can be either clustered or nonclustered; a different index entirely can be the clustered index.

