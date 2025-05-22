Writes data to the cache directly, which then gets asynchronously written to the datastore.

This can be faster for write operations but can lead to ==**data loss**== if the cache is not persisted to disk.