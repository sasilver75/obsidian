 A probe that reports whether an instance is alive or should be restarted. Keep it narrow so slow dependencies do not cause unnecessary restart loops.

> "Is this process stuck or broken beyond self-recovery?"

On failure, a process supervisor/[[Kubernetes|K8s]] kubelet might restart the process/container.

A liveness check is about restartability. It should fail only when restarting the instance is likely to help. For example, if the process deadlocked, the event loop is stuck, or the server can no longer make progress, the liveness check should fail. But if the database is temporarily down, the liveness check usually should not fail, because restarting every application instance does not fix the database.