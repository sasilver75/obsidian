





Atomic:
- Put all writes into a [[Write-Ahead Log]] (WAL) on disk before writing them to their actual location on disk. Mark it as committed in the WAL 





A set of atomically executed operations that all either succeed or fail together.
