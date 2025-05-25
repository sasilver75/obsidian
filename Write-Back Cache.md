Writes data to the cache directly, which then gets asynchronously written to the datastore.
This can be faster for write operations but can lead to ==**data loss**== if the cache is not persisted to disk.

An application writes directly to the Cache, and at some point the Cache asynchronously writes to Database. Super fast, and if you aren't reading from the Cache, you might not see the latest writes, plus list of data loss.