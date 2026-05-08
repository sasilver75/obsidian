
# Go — A Python Developer's Onboarding Guide

This is "what's actually going on" in [[Modern Go Development|Go]] from the perspective of someone who thinks in [[Python]]. It assumes you already know how to write a `for` loop and what a function is — the goal is to rewire the mental model, not teach programming.

> Companion file: [[Modern Go Development]] covers the **ecosystem** (frameworks, libraries, tooling). This file covers the **language**.

---

# 1. The Mental Model Shift

Coming from Python, here's what changes — not syntax, but the **shape of the world**:

| What you knew (Python)                       | What it becomes (Go)                                              |
| -------------------------------------------- | ----------------------------------------------------------------- |
| Code is read by an interpreter at runtime    | Code is compiled to a native binary before it runs                |
| Types are discovered as values flow through  | Types are declared/inferred at compile time and never change      |
| Exceptions bubble up through `try/except`    | Errors are ordinary return values you check explicitly            |
| `class` with inheritance, `self`, `__init__` | `struct` + methods + composition. **No classes. No inheritance.** |
| `None` is the universal "nothing"            | Every type has a **zero value**; only some types can be `nil`     |
| `asyncio` event loop, `await` everywhere     | Goroutines + channels; "blocking" code is normal                  |
| Dynamic, batteries-included                  | Static, batteries-included, **and aggressively boring**           |

**The single most important shift:** Go was designed to be read, not written. It optimizes for ten engineers maintaining the same codebase for ten years, not for the person typing today. That's why it's verbose in places Python is terse, why `gofmt` is non-negotiable, and why "clever" is a slur in Go review culture.

==Aggressively boring is the feature.==

---

# 2. Compile / Run Lifecycle

There is no REPL. There is no "run this script." Go is **compiled to a single native binary**.

```bash
go run main.go     # compile to a temp binary, run it, delete it (dev convenience)
go build ./...     # compile, write binary to ./<modulename>
go test ./...      # compile test binaries, run them
go install         # compile and put binary in $GOPATH/bin
```

**What `go run main.go` actually does** (vs `python main.py`):
- Python: hands `main.py` to the CPython interpreter, which parses + executes line by line.
- Go: compiles your code (and every transitive dependency) into a self-contained machine-code binary, then executes it. The binary has no external runtime requirements — no Go installation needed on the target machine.

This is why:
- **Startup is instant.** No interpreter warmup, no import-time module execution.
- **Errors surface at compile time.** Typos, missing fields, wrong types — caught before the program runs. You will love this.
- **There is no "import side effects" footgun** in the Python sense (no module-level code running on import — only `init()` functions, which are explicit).
- **Deployment is just copying a binary.** No `requirements.txt`, no `pip install`, no virtualenv.

```go
// hello.go
package main

import "fmt"

func main() {
    fmt.Println("hello, world")
}
```

```bash
$ go run hello.go
hello, world

$ go build hello.go && ./hello
hello, world
```

---

# 3. Variables, Types, and Zero Values

### Declaration forms

```go
var x int = 42      // explicit type
var x = 42          // type inferred (still a var declaration)
x := 42             // short form — only inside functions, not at package level
const Pi = 3.14159  // compile-time constant
```

You'll see `:=` everywhere inside functions — it's the idiomatic shorthand. `var` is mostly used at package level, when you want the zero value, or when the type isn't obvious from the right side.

### No implicit conversion — ever

```go
var i int = 42
var f float64 = i        // ❌ compile error
var f float64 = float64(i)  // ✅ explicit conversion required
```

Python silently coerces (`1 + 1.0` works); Go does not. This catches bugs but feels pedantic at first.

### Zero values — the thing that replaces `None`

When you declare a variable without initializing it, Go gives it the type's **zero value**. There's no `undefined`, no `null` for value types. ==This is a huge deal.==

| Type | Zero value |
|---|---|
| `int`, `float64`, etc. | `0` |
| `bool` | `false` |
| `string` | `""` (empty, not nil!) |
| `*T` (pointer) | `nil` |
| `[]T` (slice) | `nil` (but usable for `len`, `range`, `append`) |
| `map[K]V` | `nil` (NOT usable for writes — must `make()` first) |
| `chan T` | `nil` |
| `interface{}` / `any` | `nil` |
| `struct` | a struct with every field set to its zero value |

```go
var u User             // a fully-formed User; every field is zero-valued
fmt.Println(u.Name)    // "" (not a crash, not None — empty string)

var m map[string]int   // nil map
m["foo"] = 1           // 💥 panic: assignment to entry in nil map
m = make(map[string]int)
m["foo"] = 1           // ✅
```

**Python intuition that fails here:** in Python, `if not x:` is true for `None`, `0`, `""`, `[]`, `{}`. In Go, **there is no truthiness**. `if x` only works if `x` is already a `bool`. You write `if x != ""`, `if len(s) == 0`, `if err != nil`.

---

# 4. Functions

```go
func add(a int, b int) int {
    return a + b
}

func add(a, b int) int {  // shorthand when types match
    return a + b
}
```

### Multiple return values — the killer feature

```go
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("divide by zero")
    }
    return a / b, nil
}

result, err := divide(10, 2)
```

This is how Go avoids exceptions. Functions return `(value, error)` and the caller checks the error. You'll see this **everywhere**.

### Named returns

```go
func split(sum int) (x, y int) {
    x = sum * 4 / 9
    y = sum - x
    return  // "naked" return — returns x, y
}
```

Useful for documentation; bad for long functions (returns become invisible). Use sparingly.

### Variadic

```go
func sum(nums ...int) int {
    total := 0
    for _, n := range nums {
        total += n
    }
    return total
}

sum(1, 2, 3)            // 6
nums := []int{1, 2, 3}
sum(nums...)            // spread, like Python's *args
```

### Functions are first-class

```go
type Predicate func(int) bool

func filter(xs []int, pred Predicate) []int { /* ... */ }

evens := filter(nums, func(x int) bool { return x%2 == 0 })
```

Python `lambda` ≈ Go's anonymous `func(...) ... { ... }`.

---

# 5. Error Handling — The Big One

==This is the biggest culture shock from Python.==

There are **no exceptions**. There is no `try/except`. Functions that can fail return an `error` as their last return value, and you check it.

```go
f, err := os.Open("file.txt")
if err != nil {
    return fmt.Errorf("opening config: %w", err)  // wrap and propagate
}
defer f.Close()
```

### Why this is good (eventually)

1. **Error paths are visible in the code.** You can't accidentally let one bubble up unhandled — the compiler complains about ignored values.
2. **No surprise control flow.** A function call that returns is the function returning. There's no "and also this might unwind 30 frames into a `try/except` somewhere."
3. **Errors are data.** You can wrap, inspect, compare, and pass them around.

### The `error` type

`error` is just an interface:
```go
type error interface {
    Error() string
}
```
Anything with an `Error() string` method is an error.

### Creating errors

```go
errors.New("file not found")              // simple
fmt.Errorf("user %d not found", id)       // formatted
fmt.Errorf("getting user: %w", err)       // ⭐ wraps another error
```

The `%w` verb is special — it wraps the inner error so callers can unwrap and inspect it.

### Inspecting wrapped errors

```go
if errors.Is(err, sql.ErrNoRows) {
    // err (or anything it wraps) IS this sentinel
}

var pgErr *pgconn.PgError
if errors.As(err, &pgErr) {
    // err (or anything it wraps) IS-A *pgconn.PgError; pgErr now points to it
    fmt.Println(pgErr.Code)
}
```

- `errors.Is` — equality-style comparison (sentinel errors).
- `errors.As` — type-assertion-style unwrapping (typed errors).

### Sentinel vs typed errors

```go
// Sentinel: package-level error values to compare against
var ErrNotFound = errors.New("not found")

if errors.Is(err, ErrNotFound) { ... }

// Typed: structs that carry data
type ValidationError struct {
    Field string
    Msg   string
}
func (e *ValidationError) Error() string { return e.Field + ": " + e.Msg }

var vErr *ValidationError
if errors.As(err, &vErr) { ... }
```

### `panic` and `recover` — NOT your exceptions

`panic` exists, but it's for **truly unrecoverable** programmer errors (nil pointer deref, out-of-bounds, "this should be impossible"). Library code should almost never panic. Application code should almost never recover.

```go
defer func() {
    if r := recover(); r != nil {
        // last-resort cleanup, e.g. in an HTTP handler so one bad request
        // doesn't take down the server
    }
}()
```

If you find yourself reaching for panic/recover for control flow, you're writing Python in Go. Stop.

---

# 6. Structs, Methods, and Embedding (the "no classes" section)

Go has **no classes**. No inheritance. No `self`. No `__init__`.

What it has instead: **structs** (data) + **methods** (behavior) + **interfaces** (contracts) + **embedding** (composition).

### Structs

```go
type User struct {
    ID    int
    Name  string
    Email string
}

u := User{ID: 1, Name: "Sam", Email: "s@x.com"}  // named fields (always do this)
u := User{1, "Sam", "s@x.com"}                    // positional (fragile, avoid)
u := User{}                                       // zero-valued: {0, "", ""}
```

### Methods

A method is just a function with a **receiver** declared before the name:

```go
func (u User) Greet() string {
    return "Hello, " + u.Name
}

u.Greet()  // "Hello, Sam"
```

The `(u User)` part is the receiver — Go's equivalent of `self`, but you name it yourself and it's explicitly typed. Convention: short (1-2 letters), and consistent across all methods on a type.

### Value vs pointer receivers — important

```go
func (u User) SetName(n string)  { u.Name = n }   // pointer to a COPY — does nothing visible
func (u *User) SetName(n string) { u.Name = n }   // pointer to the original — actually mutates
```

Rules of thumb:
- **Use a pointer receiver** if the method mutates the receiver, the struct is large, or the type contains a mutex/anything that shouldn't be copied.
- **Use a value receiver** for small, immutable, value-semantic types (think `time.Time`).
- **Be consistent** — pick one for a given type and stick with it.

### Embedding (composition, not inheritance)

```go
type AdminUser struct {
    User                // embedded — note: no field name
    Permissions []string
}

a := AdminUser{User: User{Name: "Sam"}, Permissions: []string{"admin"}}
fmt.Println(a.Name)     // "Sam" — promoted from User
fmt.Println(a.Greet())  // "Hello, Sam" — User's method is promoted too
```

This is **NOT inheritance**. It's syntactic sugar for "this struct has a `User` field, and you can access its fields and methods as if they were on the outer struct." There's no virtual dispatch, no method override, no `super()`. If you define `Greet` on `AdminUser`, it shadows the embedded `User.Greet`, but `User.Greet` is still callable as `a.User.Greet()`.

**Mental model from Python:** it's not `class AdminUser(User)`. It's closer to `class AdminUser: def __init__(self): self.user = User(); ...` plus Python `__getattr__` forwarding. The `AdminUser` *has-a* `User`; it isn't *a* `User` (in the OOP sense).

### Constructors — there aren't any (officially)

By convention, you write a `New<TypeName>` function:

```go
func NewUser(name, email string) *User {
    return &User{
        Name:  name,
        Email: email,
    }
}
```

There's no special constructor syntax. `NewUser` is just a function that returns a pointer to a `User`.

---

# 7. Pointers — The Minimal Subset

==Pointers exist in Go, but they're nothing like C's pointers.== No arithmetic, no manual memory management. They're really just answering one question: **"is this thing passed by value or by reference?"**

```go
n := 5
p := &n          // p is *int — pointer to n
fmt.Println(*p)  // 5 — dereference
*p = 10          // mutate via the pointer
fmt.Println(n)   // 10
```

### Why they matter

**Everything in Go is passed by value.** When you pass a struct to a function, the function gets a *copy*. To mutate the original or avoid copying a large struct, pass a pointer:

```go
func grow(u User)  { u.Name += "!" }   // modifies a COPY; original unchanged
func grow(u *User) { u.Name += "!" }   // modifies the original

grow(&user)
```

### When to use `*T` vs `T`

- **Method receivers**: pointer if the method mutates or the type is large.
- **Function parameters**: pointer if you need to mutate, or the struct is big enough that copying matters. Otherwise a value is fine and often clearer.
- **Returning from constructors**: by convention, `NewFoo()` returns `*Foo`.
- **Optional fields in structs**: `*string` lets you distinguish "not set" (`nil`) from "set to empty string" (`""`). Sometimes necessary for JSON nullability.

### `&` and `*` cheat sheet

| Symbol | Read as |
|---|---|
| `&x` | "address of x" — gives you a pointer |
| `*p` | "value at p" — dereferences a pointer |
| `*T` | "pointer to T" — used in type declarations |

**Python intuition that fails:** in Python, every name is essentially a reference. In Go, names hold values, and you opt into reference semantics with `*T` and `&`.

---

# 8. Slices, Maps, Arrays, Strings — Not What You Think

This is the section Python devs trip on most.

### Arrays — fixed size, rarely used directly

```go
var a [3]int = [3]int{1, 2, 3}  // size is part of the type
```

You almost never write these. Go arrays are low-level building blocks; you use **slices** instead.

### Slices — the "list" you actually use

```go
s := []int{1, 2, 3}
s = append(s, 4)          // [1 2 3 4]
fmt.Println(len(s), cap(s))
s2 := s[1:3]              // slice of s, indices [1, 3) → [2, 3]
```

A slice is a small struct: **(pointer to backing array, length, capacity)**. Multiple slices can share the same backing array. ==This is the source of most slice bugs.==

```go
a := []int{1, 2, 3, 4, 5}
b := a[1:3]   // shares backing array with a
b[0] = 999
fmt.Println(a)  // [1 999 3 4 5]  ← a was mutated!
```

`append` may or may not allocate a new backing array depending on capacity. **Never assume the result of `append` shares memory with the input.** Always reassign:

```go
s = append(s, x)  // ✅ always do this
append(s, x)      // ❌ result discarded; s may or may not see the change
```

To explicitly copy:
```go
b := make([]int, len(a))
copy(b, a)
```

Or use `slices.Clone(a)` (Go 1.21+).

### Maps

```go
m := map[string]int{"a": 1, "b": 2}
m["c"] = 3
val, ok := m["d"]   // val=0, ok=false — the comma-ok idiom
delete(m, "a")
```

- A `nil` map can be **read** (returns zero value) but **not written** (panic). Always `make()` or use a literal.
- ==**Iteration order is randomized**== — Go deliberately shuffles to prevent code from relying on insertion order. (Python 3.7+ guarantees insertion order; Go does the opposite.)
- The comma-ok idiom is how you distinguish "key missing" from "key present with zero value":

```go
if val, ok := m["foo"]; ok {
    // foo is present
}
```

### Strings — immutable bytes, with rune awareness

```go
s := "hello"
s[0]              // byte 0x68 ('h'), NOT a one-character string
len(s)            // length in BYTES, not characters
s + " world"      // concatenation (allocates a new string)
```

- A `string` is an immutable sequence of bytes (typically UTF-8 encoded).
- Indexing gives you a `byte` (`uint8`), not a character.
- For Unicode-correct iteration, range over the string — it yields `(byteIndex, rune)`:

```go
for i, r := range "héllo" {
    fmt.Printf("%d: %c\n", i, r)  // r is rune (int32), each Unicode code point
}
```

- `[]byte(s)` and `string(b)` convert between strings and byte slices (allocates).
- `strings.Builder` is the equivalent of Python's `"".join(parts)` for efficient concatenation in loops.

---

# 9. Interfaces — Duck Typing, Compile-Checked

```go
type Reader interface {
    Read(p []byte) (n int, err error)
}
```

An interface is a set of method signatures. **A type satisfies an interface implicitly** — if it has the methods, it satisfies the interface. No `implements` keyword.

```go
type File struct{ /* ... */ }
func (f *File) Read(p []byte) (int, error) { /* ... */ }

// *File satisfies Reader automatically. Nothing to declare.
var r Reader = &File{}
```

This is **structural typing**: the same idea as Python's duck typing, but checked at compile time. You get the flexibility of "if it walks like a duck" plus the safety of "the compiler verified it walks like a duck."

### Small interfaces are idiomatic

The stdlib's `io.Reader` is one method. `io.Writer` is one method. `error` is one method. ==Go culture: interfaces should be tiny, defined where they're used (the consumer), not where the implementation lives.==

```go
// Consumer defines exactly what it needs:
type UserGetter interface {
    GetUser(ctx context.Context, id int) (User, error)
}

func renderProfile(g UserGetter, id int) { ... }
```

Now anything with a `GetUser` method works — your real DB, a mock, a fake, an HTTP client, whatever.

### "Accept interfaces, return structs"

```go
func NewService(db *sql.DB) *Service { ... }     // return concrete type
func (s *Service) Do(r io.Reader) error { ... }  // accept interface
```

- **Returning concrete types** lets callers see and use the full API.
- **Accepting interfaces** lets callers pass whatever they want, makes testing trivial.

### `any` (formerly `interface{}`)

The empty interface — satisfied by literally everything. Equivalent to Python's `Any`. Use sparingly; the more `any` in your code, the less the compiler is helping you.

```go
func printAnything(x any) { fmt.Println(x) }
```

To get the underlying type back, you need a **type assertion** or **type switch**:

```go
if s, ok := x.(string); ok { /* x is a string */ }

switch v := x.(type) {
case string: // v is string
case int:    // v is int
default:     // unknown
}
```

### `nil` interfaces — the famous gotcha

```go
var err error            // nil interface
var p *MyError = nil
err = p                  // err is NOT nil!
```

An interface is `nil` only if **both** its type and value are nil. Assigning a typed nil pointer to an interface gives you a non-nil interface holding a nil pointer. This burns everyone once.

---

# 10. Concurrency — Goroutines, Channels, `context`

The thing Go is famous for. ==Forget `asyncio`. Forget `await`.==

### Goroutines — cheap concurrent functions

```go
go doSomething()  // starts a goroutine; the calling code keeps going
```

A goroutine is a function running concurrently. It's not a thread — it's a [[Green Thread]] managed by the Go runtime, multiplexed onto OS threads. Starts at ~2KB stack, grows dynamically. You can have hundreds of thousands of them.

**No `async`/`await`.** Goroutines run normal blocking code. The runtime handles "if this goroutine blocks on I/O, schedule a different one on this OS thread." This is invisible to you.

```go
// Python (asyncio):
//   async def fetch(): return await db.query(...)
// Go:
//   func fetch() T { return db.Query(...) }   ← just blocks; runtime handles it
```

### Channels — typed pipes for goroutines

```go
ch := make(chan int)        // unbuffered: sends block until a receiver is ready
ch := make(chan int, 10)    // buffered: holds up to 10 before blocking

go func() { ch <- 42 }()    // send
v := <-ch                   // receive (blocks until value available)

close(ch)                   // signal no more sends
for v := range ch { ... }   // receive until closed
```

> "Don't communicate by sharing memory; share memory by communicating."

That's the slogan. Use channels to hand ownership of data between goroutines, instead of locking shared state. (Locks are also fine where they fit — see `sync` below.)

### `select` — wait on multiple channels

```go
select {
case v := <-ch1:
    handle(v)
case ch2 <- x:
    // sent x
case <-time.After(1 * time.Second):
    // timeout
case <-ctx.Done():
    return ctx.Err()
}
```

`select` blocks until **one** case can proceed, then runs that case. This is Go's primitive for "race these N things."

### `context.Context` — cancellation, deadlines, request scope

==Every function that does I/O or anything cancellable should take `ctx context.Context` as its first parameter.== This is a deeply ingrained convention.

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

result, err := db.QueryContext(ctx, "SELECT ...")
```

- `context.Background()` — root context, used at program entry points.
- `context.WithCancel(parent)` — child context with a `cancel()` function.
- `context.WithTimeout(parent, d)` — auto-cancels after duration.
- `context.WithValue(parent, k, v)` — request-scoped values (use sparingly, only for cross-cutting things like trace IDs).
- `ctx.Done()` — a channel that closes when the context is cancelled.
- `ctx.Err()` — non-nil if cancelled (`context.Canceled` or `context.DeadlineExceeded`).

When a context is cancelled, **all functions taking it should return ASAP**. This is how you avoid leaking goroutines on timeout or client-disconnect.

### `sync` — when channels are overkill

```go
var mu sync.Mutex
mu.Lock()
defer mu.Unlock()
// critical section
```

- `sync.Mutex` / `sync.RWMutex` — locks.
- `sync.WaitGroup` — wait for N goroutines to finish.
- `sync.Once` — run an init exactly once across all callers.
- `sync.Map` — concurrent map (skip until profiling tells you to use it).

### `errgroup` — concurrent fan-out with error handling

```go
import "golang.org/x/sync/errgroup"

g, ctx := errgroup.WithContext(ctx)
g.Go(func() error { return fetchA(ctx) })
g.Go(func() error { return fetchB(ctx) })
if err := g.Wait(); err != nil {
    return err
}
```

If any goroutine returns an error, the context is cancelled and `Wait()` returns the first error. This is the standard pattern for "do N things in parallel, fail if any fail."

### Goroutine leaks — the silent killer

```go
go func() {
    val := <-ch  // if nothing ever sends, this goroutine lives forever
    use(val)
}()
```

Every goroutine you start needs a clear path to termination. The two standard mechanisms: **close the channel it reads from**, or **let it observe `ctx.Done()`**:

```go
go func() {
    select {
    case val := <-ch:
        use(val)
    case <-ctx.Done():
        return
    }
}()
```

---

# 11. Packages, Modules, Visibility

### Packages

A package is a directory of `.go` files all declaring `package <name>` at the top. The package name is usually (not always) the directory name.

```go
// file: internal/user/user.go
package user

func New() *User { ... }
```

```go
// elsewhere
import "myapp/internal/user"

u := user.New()
```

### Modules — the dependency unit

A module is a tree of packages with a `go.mod` at the root.

```bash
go mod init github.com/sam/myapp     # creates go.mod
go get github.com/foo/bar            # add dependency
go mod tidy                          # clean up + add missing deps
```

`go.mod` is roughly `package.json`. `go.sum` is roughly the lockfile.

### Visibility — capitalization is everything

==This is the rule Python devs miss most often.==

- **Capitalized identifier** → exported (public): `User`, `GetName`, `MaxRetries`.
- **lowercase identifier** → unexported (package-private): `user`, `getName`, `maxRetries`.

```go
package user

type User struct {
    Name  string  // exported field
    email string  // unexported — only this package can read/write it
}

func New() *User { ... }   // exported
func validate() bool { ... } // unexported
```

There's no `private`/`public` keyword. Just capitalization. Period.

### `internal/` — compiler-enforced privacy

Code under any `internal/` directory can only be imported by code in the same module subtree. This is how you say "this is shared across our app, but no external users may import it."

### `init()` functions

```go
func init() {
    // runs once when the package is first loaded
}
```

Mostly used for registering things. Avoid putting real logic here — it's invisible from `main()`.

---

# 12. `defer`, `panic`, `recover`

### `defer` — guaranteed cleanup

```go
func process(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()  // runs when process() returns, no matter how

    // ... do stuff with f ...
    return nil
}
```

`defer` schedules a call to run when the **surrounding function** returns. Multiple defers run in **LIFO** order (last deferred, first executed).

```go
defer fmt.Println("1")
defer fmt.Println("2")
defer fmt.Println("3")
// prints: 3, 2, 1
```

Common uses: closing files, unlocking mutexes, releasing resources, logging duration.

**Gotcha:** arguments to a deferred call are evaluated **immediately**, but the call runs at return.

```go
i := 1
defer fmt.Println(i)  // captures i=1 NOW
i = 2
// prints "1" on return, not "2"
```

### `panic` — runtime explosion

A `panic` unwinds the stack, running deferred calls along the way, and crashes the program. Caused by:
- nil pointer dereferences,
- out-of-bounds slice/array access,
- writing to a nil map,
- explicit `panic("...")`,
- closing a closed channel, etc.

### `recover` — catch a panic (rarely)

Only useful inside a deferred function. Used at the top of long-running servers to keep one bad request from crashing everything:

```go
defer func() {
    if r := recover(); r != nil {
        log.Printf("recovered: %v", r)
    }
}()
```

==Don't use panic/recover for normal control flow.== That's writing Python in Go and your reviewers will hate you.

---

# 13. Generics

Added in Go 1.18 (2022). Came late, came clean.

```go
func Map[T, U any](slice []T, fn func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = fn(v)
    }
    return result
}

doubled := Map([]int{1, 2, 3}, func(x int) int { return x * 2 })
```

Type parameters in `[ ]` after the function name. Constraints go after the name (`any` = no constraint; you can also use `comparable` or define your own constraint interfaces).

```go
type Number interface {
    ~int | ~float64  // ~ allows underlying types
}

func Sum[T Number](xs []T) T {
    var total T
    for _, x := range xs {
        total += x
    }
    return total
}
```

**When to use generics:**
- Container/algorithm code that works the same for many types (`Map`, `Filter`, `Set[T]`).
- Common collection helpers (see `samber/lo`, `slices`, `maps`).

**When NOT to use generics:**
- "Just in case" abstraction. Concrete code is preferred until duplication actually appears.
- Anything that interfaces would handle naturally.

---

# 14. Memory & Performance Model

Go has a **garbage collector**. You don't manually free memory. But understanding stack vs heap matters for performance.

- **Stack-allocated** values are cheap (just a stack pointer bump). They die when the function returns.
- **Heap-allocated** values cost more (allocator + GC tracking).

The compiler does **escape analysis** to decide. If a value's lifetime might exceed the function's, it "escapes" to the heap.

```go
func newUser() *User {
    u := User{Name: "x"}
    return &u  // u escapes — its address leaves the function
}
```

You generally don't need to think about this. When you do (hot loop, profiler points to allocations), the techniques are:
- Reuse buffers (`sync.Pool`).
- Preallocate slices with `make([]T, 0, expectedCap)`.
- Pass pointers to large structs instead of copying.
- Avoid creating closures in tight loops.

Run `go build -gcflags="-m"` to see escape decisions. Run `go test -bench` + `pprof` for actual profiling.

---

# 15. Idioms — The "Go Way"

### Naming
- **Short names in small scopes**: `i`, `n`, `s`, `err`, `ctx` — fine.
- **Acronyms keep their case**: `URL`, not `Url`. `userID`, not `userId`. `HTTPServer`.
- **Receivers are short**: `func (u *User) ...`, `func (s *Server) ...`.
- **No Hungarian notation**: not `iCount`, just `count`.
- **`Get` is a smell**: `user.Name()` is preferred over `user.GetName()`.
- **Interface names ending in `-er`**: `Reader`, `Writer`, `Stringer`, `Closer`.

### Package names
- Short, lowercase, no underscores: `user`, `httputil`, `pgxpool`.
- Avoid generic names like `util`, `common`, `helpers`.
- The package name is a prefix at every call site — `user.New()` reads better than `userpackage.NewUser()`.

### Error messages
- Lowercase, no trailing punctuation: `fmt.Errorf("opening file: %w", err)`.
- Add context as you propagate: `"writing config: opening file: permission denied"` reads top-down through the call stack.

### Table-driven tests

```go
func TestAdd(t *testing.T) {
    cases := []struct{
        name    string
        a, b    int
        want    int
    }{
        {"basic", 1, 2, 3},
        {"zero", 0, 0, 0},
        {"negative", -1, 1, 0},
    }
    for _, c := range cases {
        t.Run(c.name, func(t *testing.T) {
            if got := Add(c.a, c.b); got != c.want {
                t.Errorf("Add(%d,%d)=%d, want %d", c.a, c.b, got, c.want)
            }
        })
    }
}
```

### Functional options for flexible constructors

```go
type Server struct {
    timeout time.Duration
    addr    string
}

type Option func(*Server)

func WithTimeout(d time.Duration) Option {
    return func(s *Server) { s.timeout = d }
}

func NewServer(opts ...Option) *Server {
    s := &Server{timeout: 30 * time.Second, addr: ":8080"}
    for _, opt := range opts { opt(s) }
    return s
}

srv := NewServer(WithTimeout(5 * time.Second))
```

This is Go's answer to Python's keyword arguments with defaults.

### Accept interfaces, return structs (already covered, but lives here too)

### Don't over-abstract
Three similar lines is better than a premature abstraction. Go culture is "duplicate before you abstract."

---

# 16. Gotchas That Bite Python Devs

### 1. No truthiness
```go
if s { }       // ❌ compile error if s isn't bool
if s != "" { } // ✅
if len(xs) > 0 { }
```

### 2. No comprehensions
```python
# Python
squares = [x*x for x in xs]
```
```go
// Go
squares := make([]int, len(xs))
for i, x := range xs {
    squares[i] = x*x
}
```
Or use `lo.Map` from `samber/lo`. There's no syntax sugar; you write the loop.

### 3. Maps are unordered (and randomized)
You **cannot** rely on iteration order. The runtime intentionally varies it. To iterate sorted:
```go
keys := make([]string, 0, len(m))
for k := range m {
    keys = append(keys, k)
}
sort.Strings(keys)
for _, k := range keys { ... }
```

### 4. Loop variable capture (pre-1.22)
```go
for _, x := range items {
    go func() { use(x) }()  // ⚠️ pre-1.22: all goroutines see the LAST x
}
```
Go 1.22+ fixed this — each iteration now has its own `x`. On older code, you'll see `x := x` shadowing inside the loop as a workaround.

### 5. Nil maps panic on write, but reads are fine
```go
var m map[string]int
_ = m["foo"]      // 0, ok=false — fine
m["foo"] = 1      // 💥 panic
```

### 6. Slice aliasing
Slices share backing arrays. Modifying a sub-slice mutates the parent. If you need independence, `slices.Clone()` or `copy()`.

### 7. JSON tags, not field names, are what go on the wire
```go
type User struct {
    ID   int    `json:"id"`
    Name string `json:"name,omitempty"`
}
```
Without tags, Go marshals as the (capitalized) field name, which is rarely what you want. `omitempty` skips zero values.

### 8. Zero values can mask bugs
A struct field of `int` is `0`. Was that intentional, or "not set"? Python would have `None`; Go conflates them. If you need the distinction, use `*int`, or a sentinel like `-1`, or `sql.NullInt64`-style wrappers.

### 9. `nil` interface vs interface containing nil
```go
var p *MyErr = nil
var err error = p
fmt.Println(err == nil)  // FALSE — err is a non-nil interface holding a nil *MyErr
```
Always return `error` directly (`return nil`), not a typed nil pointer assigned to one.

### 10. `range` gives a copy
```go
for _, u := range users {
    u.Name = "x"  // modifies the LOOP VARIABLE, not users[i]
}

for i := range users {
    users[i].Name = "x"  // ✅
}
```

### 11. `==` doesn't work on slices/maps/funcs
You can compare basic types, structs (if all fields are comparable), arrays, pointers. You **cannot** `==` two slices. Use `slices.Equal(a, b)` or write a loop.

### 12. No tuple destructuring assignment beyond function returns
```go
a, b = b, a   // ✅ this works (multiple assignment)
x, y := point.X, point.Y   // ✅
```
But you can't write `a, b := []int{1, 2}` — that's not a thing.

### 13. There is no `__main__` guard
Every binary has a `func main()` in `package main`. Library packages don't have one. There's no equivalent of `if __name__ == "__main__":` because the language separates "library" and "executable" at the package level.

### 14. Strings are not lists of characters
`s[0]` is a byte, not a string. `len(s)` is byte length. Iterate with `for _, r := range s` to get runes.

### 15. Channel sends/receives on nil channels block forever
Reading from or writing to a `nil` channel blocks the goroutine permanently. Useful in `select` (a `nil` case is effectively disabled), surprising everywhere else.

---

# 17. Python ↔ Go Cheat Sheet

| Concept | Python | Go |
|---|---|---|
| Run a script | `python main.py` | `go run main.go` |
| Build artifact | `.pyc` cache, source ships | Native binary, no source |
| Variable | `x = 42` | `x := 42` (inside func), `var x = 42` (top level) |
| Constant | `X = 42` (convention) | `const X = 42` (enforced) |
| Type hint | `x: int = 42` | `var x int = 42` (mandatory) |
| `None` | `None` | `nil` (only for ptr/slice/map/chan/func/interface), or zero value |
| Truthiness | `if not x:` | No truthiness; use explicit comparison |
| List | `[1, 2, 3]` | `[]int{1, 2, 3}` |
| Dict | `{"a": 1}` | `map[string]int{"a": 1}` |
| Set | `{1, 2, 3}` | `map[int]struct{}{1: {}, 2: {}, 3: {}}` (no built-in) |
| Tuple | `(1, "x")` | Anonymous struct, or just multiple returns |
| List comp | `[x*2 for x in xs]` | `for` loop, or `lo.Map` |
| f-string | `f"hi {name}"` | `fmt.Sprintf("hi %s", name)` |
| Print | `print(x)` | `fmt.Println(x)` |
| Errors | `raise/try-except` | `return ..., err` + `if err != nil` |
| Class | `class Foo:` | `type Foo struct { ... }` + methods |
| Inheritance | `class B(A):` | Embedding (`type B struct { A }`) — composition only |
| `self` | `def foo(self):` | `func (f *Foo) foo()` |
| `__init__` | `def __init__(self):` | `func NewFoo(...) *Foo` (convention) |
| Abstract base / Protocol | `abc.ABC`, `Protocol` | `interface` (implicit, structural) |
| Decorator | `@cache` | Higher-order function returning the wrapped function |
| `with` | `with open(p) as f:` | `f, err := os.Open(p); defer f.Close()` |
| Concurrency | `asyncio` + `await` | Goroutines + channels |
| Background task | `asyncio.create_task(f())` | `go f()` |
| Timeout | `asyncio.wait_for(..., 5)` | `context.WithTimeout(ctx, 5*time.Second)` |
| Mutex | `threading.Lock()` | `sync.Mutex` |
| Package install | `pip install foo` | `go get github.com/foo/bar` |
| Lockfile | `requirements.txt` / `poetry.lock` | `go.sum` |
| Virtualenv | `venv` | None — module cache is global, builds are hermetic |
| Test | `pytest` | `go test ./...` (built in) |
| Format | `black` (optional) | `gofmt` (mandatory, built in) |
| Linter | `ruff`, `pylint` | `go vet`, `golangci-lint` |
| Static type check | `mypy` | The compiler |

---

# 18. What to Actually Practice First

If you only have a weekend:

1. **`if err != nil` discipline.** Internalize that errors are values. Write a small function chain and propagate with `%w`.
2. **Build a tiny HTTP server** with `net/http` and a struct method as the handler. Get used to `http.ResponseWriter` and `*http.Request`.
3. **Write a struct with methods**, including one with a pointer receiver. Pass it to a function. Notice when changes stick and when they don't.
4. **Make a slice, append to it, sub-slice it, mutate the sub-slice.** Watch the parent change. Internalize that slices share memory.
5. **Spawn 5 goroutines via `errgroup`** that each fetch something, return errors. Cancel the parent context and watch them all clean up.
6. **Read [Effective Go](https://go.dev/doc/effective_go)** and [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments) — both short, both essential.

After that, the rest is just **reading the standard library** until the patterns become muscle memory. The stdlib is unusually well-written and is the canonical example of idiomatic Go.

---

# Further Reading
- [[Modern Go Development]] — ecosystem, frameworks, libraries, deployment
- [Effective Go](https://go.dev/doc/effective_go)
- [The Go Memory Model](https://go.dev/ref/mem)
- [Go by Example](https://gobyexample.com/) — working code for every concept
- [Tour of Go](https://go.dev/tour/) — interactive intro
