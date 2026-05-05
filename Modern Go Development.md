
# Go the Language

Go (also "Golang") is a compiled, statically-typed language from Google (2009, Rob Pike, Ken Thompson, Robert Griesemer). It was designed as a corrective to C++ — fast to compile, easy to read, built-in concurrency, no inheritance. It's ==the language of cloud infrastructure==: [[Docker]], [[Kubernetes]], [[Terraform]], [[Prometheus]], [[etcd]], [[CockroachDB]], [[HashiCorp Consul|Consul]], [[HashiCorp Vault|Vault]], Caddy, and most of the modern DevOps toolchain are Go.

**Mental model from JS/Python:** Go is what you reach for when you want Python's readability, C's performance, and Node's concurrency model — but with none of their footguns. ==It is aggressively boring by design.==

## Key Design Decisions (the "why is this weird" section)

### Statically typed, compiled
Unlike Python/JS, Go knows every variable's type at compile time. ==The binary you ship is native machine code — no interpreter==, no JIT warmup.

```go
// Python: x = 42         (type discovered at runtime)
// JS:     let x = 42     (type discovered at runtime)
// Go:     x := 42        (type inferred at compile time — still concise)
var x int = 42             // explicit
x := 42                    // shorthand (type inferred), the common form
```

### No exceptions — errors are values
This is the biggest culture shock from Python/JS. ==Go has no `try/catch`==. ==Functions return errors explicitly==.

```go
// Python: raise ValueError("bad thing")   → try/except elsewhere
// Go:     return nil, fmt.Errorf("bad thing")  → caller checks it

f, err := os.Open("file.txt")
if err != nil {
    return fmt.Errorf("opening file: %w", err)  // %w wraps for errors.Is/As
}
defer f.Close()
```

The `if err != nil` pattern ==feels repetitive at first. It grows on you==: error paths are explicit and impossible to accidentally ignore (the compiler warns on unused return values).

### Interfaces are implicit (structural typing)
No `implements` keyword. If your type has the right methods, it satisfies the interface. Like [[Duck Typing]], but checked at compile time.

```go
type Writer interface {
    Write(p []byte) (n int, err error)
}

// MyWriter satisfies Writer automatically — no declaration needed
type MyWriter struct{}
func (m MyWriter) Write(p []byte) (int, error) { ... }
```

This is more flexible than Java/C# and encourages small, composable interfaces. The standard library's `io.Reader`/`io.Writer` are the canonical example.

### Goroutines, not async/await
Go's concurrency model uses ==goroutines== ([[Green Thread]]s managed by the Go runtime) and channels for communication. You don't have to think about `async`/`await` — everything can block and the runtime handles scheduling.

```go
// Node: async function fetchUser() { await db.query(...) }
// Go:   func fetchUser() User { return db.Query(...) }  // blocks, but that's fine

go fetchUser()  // spins up a goroutine — 2KB stack, runtime-scheduled
```

### ==Structs==, not classes
Go has ==no classes==, ==no inheritance==, no `this`. You have structs with methods, and you compose behavior via embedding and interfaces.

```go
type User struct {
    ID    int
    Name  string
    Email string
}

func (u User) Greet() string {
    return "Hello, " + u.Name
}

// Embedding (composition over inheritance):
type AdminUser struct {
    User              // "inherits" User's fields and methods <<<<----------eh?
    Permissions []string
}
```

### Zero values
Every type has a ==zero value==. No `undefined`, no `null` for value types. Structs are ready to use without initialization.

```go
var i int      // 0
var s string   // ""
var b bool     // false
var p *int     // nil  (pointers can be nil)
var m map[string]int  // nil — not ready to write to, need make()
```

### `defer` — cleanup without try/finally
```go
f, _ := os.Open("file.txt")
defer f.Close()  // runs when the surrounding function returns, always
```
Above: `defer` ==Runs when the surrounding function returns, always==

### Pointers — the minimal subset
Go has ==pointers== but no pointer arithmetic. They exist mainly to distinguish =="pass by value" vs "pass by reference."==

```go
func double(x *int) { *x *= 2 }  // modifies original
n := 5
double(&n)  // n is now 10
```

### Generics (1.18+, 2022)
Go got generics late, but they're clean:

```go
func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}
```

---

# Toolchain

Go ships with an ==exceptional standard toolchain== — no npm-equivalent chaos:

```bash
go mod init myapp          # initialize a module (go.mod)
go get github.com/foo/bar  # add dependency
go build ./...             # compile
go test ./...              # run tests
go fmt ./...               # format (no config, opinionated, just run it)
go vet ./...               # static analysis
go run main.go             # compile+run for dev
```

- **`go.mod`**: Declares module name + Go version + direct dependencies. Like `package.json`.
- **`go.sum`**: Cryptographic checksums for all dependencies. Committed to git. Like a lockfile.
- **No `node_modules`**: Dependencies live in a global module cache (`~/go/pkg/mod`). Projects are lightweight.

### Linting
- **`golangci-lint`**: The meta-linter that runs many linters in parallel. The standard in CI. Configure via `.golangci.yml`.

### Live Reload (dev)
- **`air`**: Like `nodemon` for Go. Watches files, rebuilds + restarts on change.


# Project Layout

The community converged on a common layout (though Go doesn't enforce it):

```
myapp/
├── cmd/
│   └── server/
│       └── main.go     # entrypoints (one per binary)
├── internal/           # private packages (compiler-enforced, can't be imported externally)
│   ├── handler/
│   ├── service/
│   └── repository/
├── pkg/                # public packages (importable by others)
├── go.mod
└── go.sum
```

The `internal/` directory is enforced by the compiler — code there cannot be imported by external modules.


# Web Frameworks

Go's stdlib `net/http` is genuinely good and Go 1.22 added method + path param routing, so the stdlib handles a lot:

```go
mux := http.NewServeMux()
mux.HandleFunc("GET /users/{id}", handleGetUser)
http.ListenAndServe(":8080", mux)
```

Community routers/frameworks, in rough order of preference for new projects:

- **Chi**: Lightweight, composable, stdlib-compatible middleware. The clean default.
- **Gin**: Fastest adoption, batteries-included, slightly non-idiomatic. Ubiquitous in the wild.
- **Echo**: Similar shape to Gin, good OpenAPI integration.
- **Fiber**: Express-shaped, built on `fasthttp`. Fastest, but incompatible with stdlib `net/http` middleware.
- **Huma**: If you want auto-generated OpenAPI/JSON Schema from your handler types. Schema-first.
- **`net/http` stdlib**: Totally valid for simple services in Go 1.22+.

For gRPC, use **`google.golang.org/grpc`** or the newer **Connect** (`connectrpc.com`) which speaks both gRPC and HTTP/1.1.


# Database

### The Go philosophy: write SQL, generate types
Unlike Python's SQLAlchemy or Prisma, the Go community largely prefers writing SQL directly. The ORM is not the default.

- **`database/sql`**: Stdlib interface. Works with any driver. Handles connection pooling, transactions. Verbose but explicit.
- **`pgx`** (`jackc/pgx`): The best Postgres driver. Use `pgx/v5` directly or via `pgxpool` for connection pooling. Faster than the stdlib-compatible mode.
- **`sqlc`**: ==The dominant pattern for new Go services==. You write SQL queries in `.sql` files, and `sqlc` generates type-safe Go functions. No runtime reflection, compile-time guarantees.
  ```sql
  -- query.sql
  -- name: GetUser :one
  SELECT id, name, email FROM users WHERE id = $1;
  ```
  → generates `func (q *Queries) GetUser(ctx context.Context, id int64) (User, error)`
- **`sqlx`**: Thin extension over `database/sql` that adds struct scanning. A lighter alternative to sqlc if you don't want codegen.
- **`GORM`**: The ORM, Python-SQLAlchemy-shaped. Works, but generates surprising SQL, harder to debug, considered non-idiomatic by many senior Go devs. Fine for simple CRUD.

### Migrations
- **`golang-migrate`**: SQL migration files, CLI + library. The default.
- **`goose`**: Similar, also widely used.
- **`atlas`**: Schema-based, can diff and generate migrations from structs.


# Concurrency Patterns

This is where Go earns its reputation.

### Goroutines + channels
```go
ch := make(chan int, 10)  // buffered channel

go func() {
    ch <- expensiveComputation()
}()

result := <-ch  // blocks until result is ready
```

### `errgroup` — concurrent work with error propagation
The standard pattern for "do N things concurrently, fail if any fail":

```go
import "golang.org/x/sync/errgroup"

g, ctx := errgroup.WithContext(context.Background())

g.Go(func() error { return fetchUsers(ctx) })
g.Go(func() error { return fetchOrders(ctx) })

if err := g.Wait(); err != nil {
    return err  // first error returned
}
```

### `context` — cancellation and deadlines
Every long-running operation should accept `context.Context`. It's the standard way to propagate cancellation, deadlines, and request-scoped values.

```go
func fetchUser(ctx context.Context, id int) (User, error) {
    // ctx carries a deadline; db.QueryContext will cancel if exceeded
    return db.QueryContext(ctx, "SELECT ...")
}
```

### `sync` package
- `sync.Mutex` / `sync.RWMutex`: Standard locks
- `sync.WaitGroup`: Wait for N goroutines
- `sync.Once`: Run something exactly once (init patterns)
- `sync.Map`: Concurrent map (prefer a Mutex-wrapped regular map unless profiling says otherwise)

### Worker pool pattern
```go
jobs := make(chan Job, 100)
for i := 0; i < numWorkers; i++ {
    go func() {
        for j := range jobs {
            process(j)
        }
    }()
}
```


# Testing

Built-in, no framework needed:

```go
// user_test.go
func TestGreet(t *testing.T) {
    u := User{Name: "Sam"}
    if got := u.Greet(); got != "Hello, Sam" {
        t.Errorf("got %q, want %q", got, "Hello, Sam")
    }
}
```

```bash
go test ./...           # all tests
go test -race ./...     # with race detector (always run in CI)
go test -cover ./...    # coverage
```

### Table-driven tests — the Go idiom
```go
func TestAdd(t *testing.T) {
    cases := []struct {
        a, b, want int
    }{
        {1, 2, 3},
        {0, 0, 0},
        {-1, 1, 0},
    }
    for _, c := range cases {
        if got := Add(c.a, c.b); got != c.want {
            t.Errorf("Add(%d, %d) = %d, want %d", c.a, c.b, got, c.want)
        }
    }
}
```

### Key testing libraries
- **`testify`** (`stretchr/testify`): `assert`/`require` helpers. Nearly universal. `require` stops the test on failure; `assert` continues.
- **`testcontainers-go`**: Spin up a real Postgres/Redis/etc. in Docker for integration tests. The Go equivalent of Testcontainers. Should be default for DB tests.
- **`httptest`** (stdlib): `httptest.NewRecorder()` + `httptest.NewServer()` for testing HTTP handlers without a real server.
- **`gomock`** / **`mockery`**: Mock generation from interfaces.


# Configuration and CLI

### Config
- **`viper`**: Reads from env, files (YAML/TOML/JSON), flags, remote config. Often paired with `cobra`. Heavy but comprehensive.
- **`godotenv`**: `.env` file loading for dev.
- Simple env-only: `os.Getenv()` is fine for small services; `envconfig` or `env` (caarlos0/env) for struct-tag-based parsing.

### CLI
- **`cobra`**: The default for any non-trivial CLI. Powers `kubectl`, `gh`, `hugo`. Subcommands, flags, completions.
- **`kong`**: Struct-tag-based CLI parsing, less boilerplate than cobra for simpler tools.


# Logging

- **`slog`** (stdlib, Go 1.21+): ==The answer now.== Structured logging built into the standard library. JSON or text output. Leveled. Replaces the old `log` package for new code.
  ```go
  slog.Info("user created", "id", userID, "email", email)
  // → {"time":"...","level":"INFO","msg":"user created","id":42,"email":"..."}
  ```
- **`zerolog`**: Zero-allocation structured logger, very fast. Still used where performance matters.
- **`zap`** (Uber): Also high-performance, more ergonomic API than zerolog. Widely used in production services.
- Rule of thumb: `slog` for new projects unless you're benchmarking logs.


# HTTP Client

The stdlib `net/http` client is solid and production-grade. You don't need a library, but:
- **`resty`**: Fluent HTTP client, nice for calling external APIs.
- Set timeouts explicitly — the stdlib client has no default timeout, which will burn you:
  ```go
  client := &http.Client{Timeout: 10 * time.Second}
  ```


# Observability

- **[[OpenTelemetry Protocol|OpenTelemetry]] Go SDK** (`go.opentelemetry.io/otel`): The standard. Traces + metrics + logs. Vendor-neutral.
- `slog` handles structured logs natively.
- Metrics via OTel or `prometheus/client_golang` directly.


# Common Libraries Worth Knowing

| Library | Purpose |
|---|---|
| `google/uuid` | UUID generation |
| `go-playground/validator` | Struct validation via struct tags |
| `golang-jwt/jwt` | JWT parsing/signing |
| `golang-migrate/migrate` | DB migrations |
| `robfig/cron` | Cron scheduling |
| `samber/lo` | Lodash-style generics helpers (Map, Filter, etc.) |
| `go-chi/chi` | HTTP router |
| `jackc/pgx` | Postgres driver |
| `sqlc-dev/sqlc` | SQL → type-safe Go codegen |
| `stretchr/testify` | Test assertions |
| `cosmtrek/air` | Live reload for dev |
| `golangci/golangci-lint` | Meta-linter |


# Deployment

Go's killer deployment story: **a single static binary with no runtime dependencies**.

```dockerfile
# Multi-stage build — final image is ~10MB
FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o server ./cmd/server

FROM scratch              # empty base image
COPY --from=builder /app/server /server
ENTRYPOINT ["/server"]
```

- No Python interpreter, no Node runtime, no JVM — just the binary.
- **Cross-compilation** is trivial: `GOOS=linux GOARCH=arm64 go build` from a Mac works.
- Cold start is ~instant. Great for Lambda, containers, VMs.


# Patterns and Idioms

### Accept interfaces, return structs
```go
// Good: accept the interface, return the concrete type
func NewUserService(db *pgxpool.Pool) *UserService { ... }
func (s *UserService) GetUser(ctx context.Context, id int) (User, error) { ... }
```

### Options pattern for constructors with many params
```go
type ServerOption func(*Server)

func WithTimeout(d time.Duration) ServerOption {
    return func(s *Server) { s.timeout = d }
}

func NewServer(opts ...ServerOption) *Server {
    s := &Server{timeout: 30 * time.Second}  // defaults
    for _, opt := range opts { opt(s) }
    return s
}
```

### Functional options > config structs for libraries
### Table-driven tests everywhere
### Wrap errors with context: `fmt.Errorf("getting user %d: %w", id, err)`
### Use `context.Context` as the first parameter of every function that does I/O


# What's in the Standard Library (it's remarkably complete)

One of Go's strengths: the stdlib handles a lot so you don't reach for packages.
- `net/http`: Full HTTP client + server
- `encoding/json`: JSON marshal/unmarshal
- `database/sql`: DB interface
- `crypto/...`: TLS, hashing, AES, RSA
- `testing`: Test runner + benchmarks
- `sync`: Concurrency primitives
- `context`: Cancellation/deadlines
- `log/slog` (1.21+): Structured logging
- `os/exec`: Shell out to processes
- `text/template`, `html/template`: Templating

# Trends

- **`slog` replaced `logrus`/`zap` as the default logging choice** for new projects.
- **Go 1.22's routing** (`GET /users/{id}`) is eating into the router-library market for simple services.
- **`sqlc` is winning** over GORM in "how do I talk to a database" for greenfield services.
- **Generics (`samber/lo`)** finally gave Go the collection helpers it was missing; `lo.Map`, `lo.Filter` are now idiomatic.
- **`pgx` v5** is the Postgres default — the stdlib-compat shim is largely gone.
- **Go toolchain versioning** (1.21+): `go.mod` specifies the exact toolchain, reproducible builds across machines.
- The language is deliberately slow-moving and that's a feature — Go code from 2015 still compiles and runs.


# What to Learn

1. **Error handling patterns**: wrapping with `%w`, `errors.Is`, `errors.As`, sentinel errors vs typed errors
2. **Goroutines + channels + `errgroup`**: the concurrency model end-to-end
3. **`context.Context`**: why it exists, how cancellation propagates, how to use deadlines
4. **Interfaces**: how to design small, composable interfaces; why `io.Reader`/`io.Writer` are the model
5. **`sqlc` + `pgx`**: the modern DB access story
6. **Testing**: table-driven tests, `testify`, `testcontainers-go`
7. **The stdlib**: `net/http`, `encoding/json`, `sync`, `slog` — Go rewards knowing stdlib depth
8. **Profiling**: `go tool pprof` — CPU and memory profiling is built in
9. **Build and deploy**: single binary, multi-stage Docker, cross-compilation


# Go vs Python/JS — Quick Reference

| Concept | Python/JS | Go |
|---|---|---|
| Errors | `raise`/`try-except`, `throw`/`try-catch` | `return nil, err` + `if err != nil` |
| Concurrency | `asyncio`/`async-await`, Promises | goroutines + channels + `errgroup` |
| Types | Dynamic / gradual (TS) | Static, inferred |
| Classes | `class` with inheritance | Structs + methods + interfaces (no inheritance) |
| Null | `None`, `undefined` | `nil` (only for pointers/interfaces/slices/maps/channels) |
| Packages | `pip`/`npm` | `go get` + `go.mod` |
| Formatting | `black`/`prettier` (optional) | `gofmt` (non-negotiable, built in) |
| Runtime | CPython/V8/JIT | Compiled native binary |
| Deploy | Needs interpreter | Single static binary |
