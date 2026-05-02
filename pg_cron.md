A [[PostgreSQL|Postgres]] extension. 

`pg_cron` is a Postgres extension that ==runs scheduled jobs INSIDE the database, using standard cron syntax.== 
- Instead of an external scheduler (`systemd` timer, Github Actions, Vercel Cron) calling into your DB, the DB schedules and executes the SQL itself)

### How it Works:
- Runs as a background worker in the Postgres process.
- You schedule jobs by inserting into `cron.schedule(...)` and a worker process makes up on the cron tick to run them.
- Job history lands in `cron.job_run_details`

## Where it shines
- ==Periodic maintenance==: vacuum analyze, reindex, partition rotation, refreshing materialized views
- Data lifecycle: Deleting expired rows, archiving, soft-delete cleanup
- ==Aggregations/rollups: Hourly stats tables computed from raw events==
- Triggering outbound work via LISTEN/NOTIFY or by writing to a queue table than app worker drains

## Where it doesn't fit
- Long-running jobs: A job that takes 20 minutes holds a connection and a worker slot the whole time; Use pg_cron to ENQUEUE work, not to do heavy work.
- App-level business logic that needs observability
- HTTP calls/external side effects: Possible via `pg_net` or `plpython3u`, but you're piling extensions on extensions and writing shit where you shouldn't.
- Replicas: ==Jobs run on the primary only. After a failover, the new primary picks them up, but if you have multiple writers (rare in Postgres) or read replicas you expect to schedule against, it won't work.==


```SQL
-- Add the extension
CREATE EXTENSION pg_cron;

-- Run every night at 3am
SELECT cron.schedule(
'nightly-cleanup',
'0 3 * * *',
$$ DELETE FROM sessions WHERE created_at < now() - interval '30 days' $$
);

-- Every 5 minutes
SELECT cron.schedule('refresh-mv', '*/5 * * * *', 'REFRESH MATERIALIZED VIEW stats');

-- Sub-minute (added in 1.5): every 30 seconds
SELECT cron.schedule('heartbeat', '30 seconds', 'SELECT 1');

-- Unschedule by name or job id:
SELECT cron.unschedule('nightly-cleanup');
```

By default, `pg_cron` only runs jobs in the database where it's installed (typically `postgres`). To run a job in another database, use `cron.schedule_in_database(...)`:   ==This trips people up. People install the extension, schedule jobs, and the jobs silently never touch the right database.==
```SQL
  SELECT cron.schedule_in_database(
    'app-cleanup',
    '0 * * * *',
    'DELETE FROM tmp WHERE ...',
    'app_db'
  );
```


