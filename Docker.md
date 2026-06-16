
Docker is a platform for building, packaging, sharing, and running applications in [[Container]]s.
- A container is an isolated process environment that contains an application *plus* the files, libraries, environment variables, and configuration needed to run it, making software easier to run consistently across different machines.
- A container *is not a full [[Virtual Machine]]*. It is usually a normal host operating system process with an isolated view of the system.

> An image is like a frozen application package. A container is what happens when Docker starts that package as a running process.

> Docker is a toolchain for turning an application plus its runtime environment into an image, and then running that image as an isolated process called a container.

> Dockerfiles are good at answering "What exactly must exist inside the application's runtime environment," but doesn't answer questions like "Which database should this app use, how many replicas should I run, which secrets should production injection, and how should traffic be routed?" these are orchestration/infrastructure questions, handled by platforms like [[Docker Compose]], [[Kubernetes]], [[Fly.io]], [[Railway]], [[HashiCorp Nomad|Nomad]], [[Amazon Elastic Container Service|ECS]], etc.

Terms:
- Dockerfile: A text file describing how to build an application environment.
- Image: A packaged, immutable template created from a Dockerfile.
- Container: A running instance of an image.
- Registry: A place to store and download images, such as Docker Hub or a private registry.
- Docker Engine: The background service that builds images and run containers.
	- ==BuildKit== is Docker's modern image build engine, which takes a build definition (Dockerfile), turns it into a build graph, executes that graph efficiently, and produces an image or another build output.
	- The older docker builder treated Dockerfiles as a linear sequence of steps, which worked, but was less efficient at caching, often sent too build build context, and executed more work than necessary.
		- BuildKit improves the build process by treating it as a dependency graph, rather than merely as a text file executed line by line.
- [[Docker Compose]]: A tool for running multiple related containers together, such as an app plus a database.

How it works: On Linux, Docker relies mainly on operating system features.
- Namespaces: Give the containers its own view of processes, networking, mounts, users, and hostnames... isolating what the process can see, in these respects.
- Control groups, often called [[Linux Control Group|cgroup]]s: Limit and measure CPU, memory, and other resource usage.
- Union filesystem/layered filesystem: Lets images be built from reusable filesystem layers. 
- Copy-on-write storage: Lets containers start from shared image layers, then write changes separately.  Let many containers share the same image layers while each container gets its own writable layer.


```dockerfile
# Base Image: Start from an existing image with Node.js 22 installed.
FROM node:22

# Create and use /app as the working directory.
WORKDIR /app

# Copy dependency files.
COPY package*.json ./
# Install dependencies.
RUN npm install

# Copy the application code.
COPY . .

# Run npm start when the container starts.
CMD ["npm", "start"]
```
Then you'd run it like
```bash
# Start a build: docker buildx build [OPTIONS] PATH | URL | -
# -t is "tag", an image identifier.
docker build -t my-web-app .

# Create and run a new container from an image: docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
# -p is publish-list: Publish a container's ports to the host. Maps a port on your machine to a port inside the container.
docker run -p 3000:3000 my-web-app
```

When you run `docker run nginx`:
- Finds or downloads the `nginx` image.
- Creates a new writeable container on top of the image.
- Sets up namespaces and resource controls.
- Creates networking for the container.
- Starts the configured process from the image.
- Streams logs and tracks the process lifecycle.


# What a Dockerfile can express:
- `FROM`: Choose a base image and start a build stage.
- `ARG`: Define build-time variables.
- `ENV`: Define environment variables *baked into the image*.
- `WORKDIR`: Set the working directory for later instructions.
- `COPY`: Copy files from the build context, or another stage.
- `ADD`: Like `COPY`, but with extra behavior, such as allowing remote URLs and archive extraction. Usually avoid this instruction, unless you actually need this behavior.
- `RUN`: Execute *build-time commands* and commit the result as a image layer.
- `CMD`: Provide the default command or arguments *when a container starts*.
- `ENTRYPOINT`: Define the main executable for the container.
- `USER`: Set the user used for later build instructions and container runtime.
- `EXPOSE`: Document which port the containerized process is expected to listen on.
- `VOLUME`: Declare paths intended to use external persistent storage.
- `LABEL`: Add metadata to the image.
- `HEALTHCHECK`: Define a command Docker can use to classify the container as healthy or unhealthy.
- `SHELL`: Change the default shell used for shell-form commands.
- `STOPSIGNAL`: Define which signal Docker sends to stop the container.
- `ONBUILD`: Register instructions that run later when *another Dockerfile* uses this image as a *base image*.


> Build configuration into the image only when it is truly part of the application artifact. ==Inject environment-specific configuration at runtime==.
- Belongs in the image:
	- Node.js version, installed dependencies, compiled application code, startup command, default port, non-root user
- Does not belong in the image:
	- Production database password, stripe secret key, deployment-specific hostname, customer-specific configuration, cloud credentials
This means ==the same image should be portable across environments!==
```
same image -> dev
same image -> staging
same image -> production
```



Here's a realistic Dockerfile for a cloud-deployable Node.js TypeScript web API:
- Assumptions: Has TS source code in `src/`, builds to `dist/`, listens on `PORT`, has a `/healthz` endpoint, reads external configuration from environment variables, should run as a non-root user, and should not ship development dependencies into production.

```dockerfile
# syntax=docker/dockerfile:1.7

# Build-time variable used to choose the Node.js base image version.
# This can be overridden with:
# docker build --build-arg NODE_VERSION=22.11.0 .
ARG NODE_VERSION=22.11.0

# Start a shared base stage using the official Node.js image.
# bookworm-slim is Debian Bookworm with a smaller package set than the full image.
FROM node:${NODE_VERSION}-bookworm-slim AS base

# Set the default directory for later Dockerfile instructions and for runtime commands.
WORKDIR /app

# Set runtime defaults baked into the image.
# These can still be overridden with docker run -e or by the deployment platform.
ENV NODE_ENV=production
ENV PORT=3000

# Install small runtime utilities needed by the container itself.
# curl is used by the HEALTHCHECK, and ca-certificates allows HTTPS requests to verify certificates.
# The apt package lists are removed afterward to keep the image smaller.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Start a dependency-install stage from the shared base stage.
FROM base AS deps

# Temporarily switch to development mode so npm installs both production and development dependencies.
# Development dependencies are needed for TypeScript compilation, bundling, tests, or build tools.
ENV NODE_ENV=development

# Copy only dependency manifest files first.
# This improves Docker layer caching: npm ci only reruns when package.json or package-lock.json changes.
COPY package.json package-lock.json ./

# Install dependencies exactly from package-lock.json.
# npm ci is preferred for reproducible builds because it does not rewrite the lockfile.
RUN npm ci

# Start a build stage that includes installed dependencies.
FROM deps AS build

# Copy build configuration and source code after dependency installation.
# Source changes invalidate this layer and later layers, but not the cached npm ci layer above.
COPY tsconfig.json ./
COPY src ./src

# Compile the application, usually from TypeScript source into JavaScript output under dist/.
RUN npm run build

# Remove development dependencies after the build output has been created.
# The resulting node_modules directory contains only runtime dependencies.
RUN npm prune --omit=dev

# Start the final runtime image from the smaller shared base stage.
# Previous build stages are not included unless files are explicitly copied from them.
FROM base AS runtime

# Create a dedicated unprivileged operating-system user for the application.
# Running as a non-root user reduces the impact of application or dependency vulnerabilities.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

# Copy package metadata into the runtime image.
# --chown ensures the unprivileged app user owns the files.
COPY --from=build --chown=app:app /app/package.json ./package.json
COPY --from=build --chown=app:app /app/package-lock.json ./package-lock.json

# Copy the pruned runtime dependencies from the build stage.
COPY --from=build --chown=app:app /app/node_modules ./node_modules

# Copy the compiled application output from the build stage.
COPY --from=build --chown=app:app /app/dist ./dist

# Switch to the unprivileged user for the default runtime process.
USER app

# Document that the application expects to listen on port 3000.
# EXPOSE does not publish the port by itself; docker run -p or the deployment platform does that.
EXPOSE 3000

# Define a container health check.
# Docker runs this command periodically inside the container.
# A successful HTTP response from /healthz means the container is considered healthy.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl --fail http://127.0.0.1:${PORT}/healthz || exit 1

# Define the default command run when the container starts.
# This can be overridden by docker run arguments or by the deployment platform.
CMD ["node", "dist/server.js"]
```


# Docker CLI Usage

The main flow for a typical web application is going to be something like:
```bash
# Build docker image from current dir's dockerfile, image reffered ti as "my-api:local"
# The "my-api:local" reference is an image/repo name (my-api) and a tag (local)
docker build -t my-api:local .
# Run a new container from the "my-api:local" image. "--rm" deletes the container automatically after it exits. "-p 3000:3000" maps host port 3000 to container port 3000. "--env-file .env" loads runtime environment variables from the .env files. With "name", we're running the container as my-api.
docker run -d --rm --name my-api -p 3000:3000 --env-file .env my-api:local

# Now let's show logs from a running or stopped container
docker logs my-api

# Open an interactive shell inside a running container. -i=interactive, -t=pseudo-TTY
# This is useful for debugging files, environment variables, installed package, state
docker exec -it my-api sh

# Stop a running container gracefully; Docker sends the container's stop signal (SIGTERM)
docker stop my-api
```

Modern docker commands are usually shaped like: 
- `docker <object> <action>`
```
docker image ls
docker container ls
docker container stop my-api
docker volume ls
docker network ls
```

There are older shorthand commands like `docker ps`, `docker images`, `docker run`, `docker ps`, etc... which are still extremely common. In common, people do use both.

Images
```bash
# List local images
docker image ls
# Remove an image called "my-api:local"
docker image rm my-api:local
# Pull an image from a registry
docker pull node:22-bookworm-slim
# Inspect image metadata
docker image inspect my-api:local

```
Important distinction about image references
```
|-Reference-|
my-api:local
│      │
│      └── tag
└───────── image name / repository name
```


Containers
```bash
# Run a container in the foreground
# Note that docker run is a convenience command: it creates a container, startsa  container, and optionally attachesterminal/log output.
docker run --rm -p 3000:3000 my-api:local
# Run a container in the background! (-d: detached)
docker run -d --name my-api -p 3000:3000 my-api:local
# List running containers
docker ps
docker container ls
# Stop a container
docker stop my-api
docker container stop my-api
# Remove a stopped container
docker rm my-api
```

Useful `docker run` flags:
- `--rm`: Delete the container automatically when it exits
- `-d`: Run detached, in the background
- `--name my-api`: Give the container a stable name
- `-p 3000:3000`: Publish host port 3000 to container port 3000
- `-e PORT=3000`: Set one environment variable
- `--env-file .env`: Load environment variables from a file
- `-it`: Allocate an interactive terminal
- `--entrypoint sh`: Override the image's entrypoint
- `--user 1000:1000`: Run as a specific user ID/group ID
- `--workdir /app`: Set the working directory
- `--mount ...`: Mount host files, directories, or Docker volumes
- `--network ...`: Choose the container network
- `--restart unless-stopped`: Configure restart behavior

Note on Ports:
```bash
# Maps localhost:8080 on your machine to port 3000 inside container
# HOST_PORT:CONTAINER_PORT
docker run -p 8080:3000 my-api:local
```

Environment variables:
```bash
# Pass one variable
docker run -e NODE_ENV=production my-api:local
# Pass several varaibles from a afile
docker run --env .env my-api:local
```
with example `.env`:
```
PORT=3000
DATABASE_URL=postgres://user:password@host:5432/app
LOG_LEVEL=info
```

Logs:
```bash
# Show logs
docker logs my-api
# Follow logs
docker logs -f my-api
# Show recent logs
docker logs --tail 100 my-api
```
The normal convention in Docker is that application logs should go to standard output and standard error.

Executing commands in a container:
```bash
# Start a shell inside a running container
docker exec -it my-api sh
docker exec -it my-api bash # If the image has Bash installed
```





__________________

Running software requires not just your code, but also the right version of every:
- Tool
- Library
- System dependency

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







