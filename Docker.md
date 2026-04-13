
Running software requires not just your code, but also the right version of every:
- Tool
- Library
- System depenency

... that your code relies on. Without Docker, two developers will have the "it works on my machine" problem, using different versions of Python, Postgres, or some arbitrary C library intsalled.

==Docker packages your code and its entire environment into a single artifact that runs the same way everywhere, whether on your laptop or in production.==
- We can also *isolate* services: We can run [[PostgreSQL|Postgres]], [[Redis]], application code, and a [[Tile Server]] all in different isolated containers on the same machine.


# Core Concepts

### Dockerfile
- A ==recipe for building an image== -- a sequence of instructions that start from a base image and layer changes on top.
	- Commonly starts from an existing image, sets working directory int he container, runs shell commands to install dependencies, copies files from your machine into the image, etc.
	- Each RUN/COPY command creates a new layer in the image.
- Key instructions:
	- FROM: Sets the base image. Each Dockerfile starts with this.
	- WORKDIR: Sets the working directory for all subsequent instructions.
	- RUN: Executes a shell command during the build.
	- COPY: Copies files from host into the image.
	- ENV: Sets environment variables baked into the image.
	- EXPOSE: Documents which port the container listens on 
	- CMD: Default command when the container starts; overwritten by "command:" in Docker Compose.
	- ENTRYPOINT: Like CMD, but harder to override... used when the container IS the command.

### Image
- A docker image is a ==read-only snapshot of a filesystem==, including an OS base, installed software, configuration files, and application code.
- These are built from Dockerfiles. 
- They don't run, they're just templates.

### Container
- A container is a ==running instance of an image==: An ==isolated process with its own filesystem, network, and process space.==
- You can run many containers from the same image simultaneously.

### Layer Caching
- Every `RUN`, `COPY`, and `ADD` instruction creates an ==immutable layer== stacked on top of the previous one.
- Docker caches each layer.
- When you rebuild, Docker starts from the ==first layer that changed==, and re-executes everything below it; unchanged layers are served from cache instantly.
	- This means that ==instruction order matters==!

Here's how:
```dockerfile
# BAD: code changes always invalidate the dependency install layer
COPY . .               # ← changes on every code edit
RUN uv sync --no-dev   # ← re-runs every time (slow!)

# GOOD: dependency install is cached until pyproject.toml changes
COPY pyproject.toml .  # ← only changes when you add/remove packages
RUN uv sync --no-dev   # ← cached unless pyproject.toml changed
COPY . .               # ← code changes don't affect the layer above
```


### Networking: Port Mapping
- Docker containers are by default isolated from outside world.
- ==Port mapping== publishes as container port to your host machine so you can reach it from your browser or terminal:
```yaml
ports:
  - "5432:5432"   # host_port:container_port
```

### Networking: Container-to-Container Networking
- When containers are on the same Docker network (Docker Compose creates one automatically), ==they reach eachother by service name, not by localhost==.
```
# From inside the api container:
DATABASE_URL = postgresql://la_obs:password@db:5432/la_observatory
                                              ^^
                                        db is service name, not localhost
```
Above: `localhost` inside a container refers to that container itself, NOT to the host machine or to other containers! This trips people up constantly.

### Volumes
- Containers themselves are *ephemeral*: When a container is removed, everything written to its filesystem is gone.
- ==Volumes are how you persist data across container restarts!==
- ==Named Volumes==: Managed by Docker and stored in Docker's own data directory on your host.
```yaml
# This is av olume mount on the service, saying "attach the named volume ;postgres_data' to the container at the path /var/lib/postgresql/data"; that path is where Postgres writes all its data files. Anything Postgres writes there goes into the volume instead of hte container's ephemeral filesystem.
# "Where to attach"
volumes:
  - postgres_data:/var/lib/postgresql/data  # host path : container path

# This is the top-level declaration, registering postgres_data as a named volume managed by Docker; without this, the mount above would fail, and Dockoer wouldn't know what postgres_data refers to.
# The value after the colon is empty, meaning "use all defaults" Docker picks the storage location on your host.
# "What it is"
# Declared at the bottom of docker-compose.yml
volumes:
  postgres_data:
```
- Above: The postgres_data container will survive a `docker compose down`: You have to destroy it with `docker compose down -v`, if you want to.

### Bind Mounts
- Maps a directory on your local machine into the container: Changes are reflected immediately in both directions, with no rebuild needed.
```yaml
# Note that .. works the same as in any unix path
volumes:
  - ../backend:/app   # host path : container path
```
- This is how reload works: You edit `backend/app/main.py` on your Mac, and the change appears inside the container at `/app/app/main.py,`, and uvicorn's `--reload` flag detects it and restarts the server.

### Docker Compose
- A tool for defining and running ==multi-container applications== -- instead of  running `docker run` commands manually for each service, describe everything inside a `docker-compose.yml` file.


```yaml
services:
  service_name:
    image: or build:    # use an existing image, or build one from a Dockerfile
    environment:        # environment variables injected into the container
    ports:              # host:container port mappings
    volumes:            # named volumes or bind mounts
    depends_on:         # start order and health dependencies
    command:            # override the CMD from the Dockerfile
    healthcheck:        # how to test if the service is ready

volumes:                # declare named volumes
```
Above: `build` vs `image`:
- Use a pre-built image from Docker Hub, or build from a local Dockerfile
```yaml
# Use a pre-built image from Docker Hub
redis:
  image: redis:7-alpine

# Build from a local Dockerfile
api:
  build:
    context: ../backend    # directory containing the Dockerfile
    dockerfile: Dockerfile
```


### Hot Reload Without Rebuilding

In my La Observatory project, the `api` and `worker` services bind-mount their source directories:

```yaml
volumes:
  - ../backend:/app
```

Combined with `uvicorn --reload` (api) and Celery's auto-reload (worker), code changes take effect immediately without a `docker compose up --build`.

The `--build` flag is only needed when you change a `Dockerfile` or `pyproject.toml`.







