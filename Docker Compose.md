
In [[Docker Compose]], instead of just running containers, we define and run ==multi-container *applications*== defined in a `compose.yaml` file.
- A realistic application is rarely just one container; a local version of a cloud-style web app might need: a web API container, PostgreSQL container, Redis container, background worker container, database migration command, shared network, persistent database volume, environmental variables, port mappings, health checks.
- While the raw [[Docker]] CLI offers imperative commands, Docker Compose allows you to declaratively state your local application topology: What services should exist, the images that they should build/pull, the ports they publish, the environment variables that should be present, the volumes that should be mounted, and how they should share the network.

> Docker's declarative configuration layer for running multiple related containers together with shared networks, volumes, environment variables, build settings, and startup rules.

```yaml
services:
  app:
    ...
  database:
    ...
  redis:
    ...
```
This can start both the application and the ancillary services together, in the appropriate order.
```bash
docker compose up
```

Docker Compose mostly organizes five Docker concepts:
- `services`: Long running processes such as `api`, `db`, `redi`, `worker`. A service is a template for *one or more containers!*
- `build`:  Build an image from a Dockerfile
- `image`: Use an existing image or assign a name to a built image
- `volumes`: Persistent or mounted storage
- `networks`: Communication boundaries between containers

### `docker compose` vs `docker-compose`
Modern docker actually uses ==docker compose==, which is implemented as a Docker CLI plugin, while older tutorials often use `docker-compose`, a standalone Compose command.
```bash
docker compose # new, Docker Compose v2, implemented as a `docker` CLI plugin
docker-compose # Old, standalone Compose command
```


Here's a realistic Compose file:
```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
      args:
        NODE_VERSION: "22.11.0"
    image: my-api:local
    ports:
      - "3000:3000"
    environment:
      NODE_ENV: production
      PORT: "3000"
      DATABASE_URL: postgres://app:app_password@postgres:5432/app
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      migrate:
        condition: service_completed_successfully
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://127.0.0.1:3000/healthz"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 20s
    restart: unless-stopped

  worker:
    image: my-api:local
    command: ["node", "dist/worker.js"]
    environment:
      NODE_ENV: production
      DATABASE_URL: postgres://app:app_password@postgres:5432/app
      REDIS_URL: redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      migrate:
        condition: service_completed_successfully
    restart: unless-stopped

  migrate:
    image: my-api:local
    command: ["node", "dist/migrate.js"]
    environment:
      NODE_ENV: production
      DATABASE_URL: postgres://app:app_password@postgres:5432/app
    depends_on:
      postgres:
        condition: service_healthy
    restart: "no"

  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: app
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app_password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d app"]
      interval: 5s
      timeout: 3s
      retries: 10
    restart: unless-stopped

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  postgres-data:
  redis-data:
```
This is a local orchestration file; it's not yet "production infrastructure," but expresses many of the same relationships that a production platform must understand.

You can see that commonly under a `service`, we seem to have:
- `build`: Tells Compose how to build the image.
- `image`: This gives the built image from `build` a name, or can otherwise mean "pull this image," when no `build` is provided.
- `ports`: This publishes a container port to the host machine, with the `HOST_PORT:CONTAINER_PORT` format.
	- The `api` service can reach PostgreSQL through `postgres:5432`, even without publishing PostgreSQL to the host.
- `environment`: Runtime environment variables injected into the container. 
	- Notice for some items like `DATABASE_URL`, the value is `postgres://app:app_password@postgres:5432/app`, and `REDIS_URL` is `redis://redis:6379`. The hostnames `postgres` and `redis` are service names. Compose provides [[Domain Name Service|DNS]] resolution between services on the same network.
- `depends_on`: Controls startup ordering. Can wait until container has started, or until a dependency's configured health check reports healthy.
- `volumes`: Service-level volume mounts, together with the top-level `volumes` declaration. The top-level volume declaration might declare a volume called `postgres-data`, and then inside a service definition's volume section, we do something like `postgres-data:/var/lib/postgresql/data`, which mounts it into the (e.g.) postgres container at `/var/lib/postgresql/data`.
- `healthcheck`: Asks "Is this containerized service actually usable?" Not always just "is the process still running?" For an API, a health check might call `/healthz`, and for PostgreSQL, a health check might run `pg_isready`. 
- `restart`: This tells Docker what to do if a container exits. Options include `"no"`, `always`, `unless-stopped`, `on-failure`. 
	- For a migration job, `"no"` is usually correct, while for a web server, locally, it might be reasonable to use `unless-stopped`.


# Using the Compose CLI

```bash
# Start everythign in the foreground
docker compose up
# Start everything in the background
docker compose up -d
# Build images before starting
docker compose up --build

# Stop and remove containers/networks
docker compose down
# Stop and remove containers/networks/volumes
docker compose down --volumes

# List compose containers
docker compose ps
# Show logs
docker compose logs
# Follow logs for one service
docker compose logs -f api
# Run a command in an already-running service container
docker compose exec api sh  # ((For the `docker compose exec` command, Compose allocates an interative terminal by default, so no "-it" needed))
# Create a new one-off `api` contaienr and run sh there (then remove the container on stop)
docker compose run --rm api sh

# Build only
docker compose build
# Pull images
docker compose pull
# Stop, without removing containers
docker compose stop
# Start stopped containers
docker compose start
```


Compose Networking
- By default, Compose creates a project-specific network. Every service joins that network unless configured otherwise.
- This gives service-name DNS:
	- `api` can reach `postgres` at `postgres:5432`
	- `api` can reach `redis` at `redis:6379`
	- `worker` can reach `postgres` at `postgres:5432`

Environment Files
- Compose variable substitution (this env file configures Compose itself!)
Given `.env`
```env
NODE_VERSION=22.11.0
API_PORT=3000
```
Compose can then substitute values into the YAML
```yaml
services:
  api:
    build:
      args:
        NODE_VERSION: ${NODE_VERSION}
    ports:
      - "${API_PORT}:3000"
```
- You can also use container environment variables (these pass environment variables into the container)
```yaml
services:
  api:
    environment:
      DATABASE_URL: postgres://app:app_password@postgres:5432/app

services:
  api:
    env_file:
      - .env.api
```

Common Development pattern:
For local development, you can often use a bind mount for source code:
```yaml
services:
  api:
    build:
      context: .
      target: deps
    command: ["npm", "run", "dev"]
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgres://app:app_password@postgres:5432/app
    volumes:
      - .:/app
      - node-modules:/app/node_modules

volumes:
  node-modules:
```
Above: The `.:/app` part makes local source changes appear inside the container, while the named volume `node-modules:/app/node_modules` prevents the host's own `node-modules` situation from clobbering the container's dependency directory. 
- ==This is important to understand==. Here, `node-modules` is actually the named volume (created if it doesn't exist) `node-modules`, which persists between runs. It is NOT the localhost's `./node_modules` directory. So first we do `./app`, and THEN we sort of overwrite a small part of that, saying ("Actually, we'll have a little carveout for the container's app/node_modules, which will come from this named volume `node-modules`, which will be created if it doesn't already exist.)