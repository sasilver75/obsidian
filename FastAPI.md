
FastAPI is a Python web framework for building HTTP APIs. Its core idea is simple: write normal Python functions with type hints, attach them to HTTP routes, and let FastAPI use those types to validate input, serialize output, and generate OpenAPI documentation automatically.

This note is a practical, first-principles tutorial. It assumes Python 3.10+ and Pydantic v2 style APIs. Official FastAPI documentation was checked on 2026-06-03.

## Table of Contents

- [Mental Model](#mental-model)
- [Setup](#setup)
- [The Smallest App](#the-smallest-app)
- [HTTP and Path Operations](#http-and-path-operations)
- [Type Hints and Validation](#type-hints-and-validation)
- [Request Data](#request-data)
- [Response Data](#response-data)
- [Errors](#errors)
- [Dependencies](#dependencies)
- [Security and Authentication](#security-and-authentication)
- [Middleware, CORS, and Cross-Cutting Behavior](#middleware-cors-and-cross-cutting-behavior)
- [Settings and Lifespan](#settings-and-lifespan)
- [Databases](#databases)
- [Bigger Applications](#bigger-applications)
- [Complete Application Walkthrough](#complete-application-walkthrough)
- [Testing](#testing)
- [Advanced Interfaces](#advanced-interfaces)
- [Deployment](#deployment)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Sources](#sources)

## Mental Model

FastAPI is built from three major pieces:

- Python type hints describe what your API accepts and returns.
- [[Pydantic]] validates and serializes data based on those types.
- Starlette provides the underlying ASGI web toolkit: routing, requests, responses, middleware, background tasks, WebSockets, static files, and test clients.

ASGI means "Asynchronous Server Gateway Interface". It is the Python standard interface that lets an async web server, such as Uvicorn, talk to an async application, such as FastAPI. This is the async equivalent of WSGI, which older frameworks like Flask and Django originally used.

A request travels roughly like this:

1. A client sends an HTTP request.
2. Uvicorn receives the network request.
3. Uvicorn calls the FastAPI ASGI application.
4. FastAPI finds the matching route.
5. FastAPI resolves dependencies.
6. FastAPI validates path, query, header, cookie, body, form, and file inputs.
7. Your path operation function runs.
8. FastAPI validates or filters the response if a response model is declared.
9. Middleware can modify the response.
10. Uvicorn sends the response to the client.

The most important vocabulary:

- App: the `FastAPI()` object.
- Path: the URL path, such as `/users/{user_id}`.
- Operation: the HTTP method, such as `GET`, `POST`, `PATCH`, or `DELETE`.
- Path operation: a method plus a path.
- Path operation decorator: `@app.get(...)`, `@app.post(...)`, and similar decorators.
- Path operation function: the Python function called for that route.
- Dependency: a function, class, or callable FastAPI runs before a route to provide a value or enforce behavior.
- Schema: a machine-readable description of data or the whole API.
- OpenAPI: the standard API schema FastAPI generates.

## Setup

Create a project and virtual environment:

```bash
mkdir fastapi-demo
cd fastapi-demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install FastAPI:

```bash
pip install "fastapi[standard]"
```

The `standard` extra installs FastAPI plus common runtime dependencies, including the FastAPI CLI, Uvicorn, standard Pydantic extras, form parsing support, and other packages commonly needed for normal API development.

A minimal `pyproject.toml` for an application can look like this:

```toml
[project]
name = "fastapi-demo"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fastapi[standard]",
]

[tool.fastapi]
entrypoint = "main:app"
```

Common commands:

```bash
fastapi dev
fastapi dev main.py
fastapi run main.py
uvicorn main:app --reload
uvicorn main:app --host 0.0.0.0 --port 8000
```

Use `fastapi dev` or `uvicorn --reload` during development. Use `fastapi run` or a managed Uvicorn/Gunicorn setup in production. The development reloader is useful locally but should not be used as a production process manager.

## The Smallest App

Create `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World"}
```

Run it:

```bash
fastapi dev main.py
```

Open:

- `http://127.0.0.1:8000/` for the endpoint.
- `http://127.0.0.1:8000/docs` for Swagger UI.
- `http://127.0.0.1:8000/redoc` for ReDoc.
- `http://127.0.0.1:8000/openapi.json` for the generated OpenAPI schema.

This one function already shows the FastAPI pattern:

```python
@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello, World"}
```

`@app.get("/")` registers the function for `GET /`. The function return value is converted to JSON. The return annotation helps editors and documentation, but the decorator and function body are what make the route work.

## HTTP and Path Operations

HTTP is a request-response protocol. A request usually contains:

- Method: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, etc.
- Path: `/users/123`.
- Query string: `?limit=20&offset=40`.
- Headers: metadata such as `Authorization` or `Content-Type`.
- Cookies: browser-managed key-value data.
- Body: JSON, form data, file upload, or another payload.

A response usually contains:

- Status code: `200`, `201`, `204`, `400`, `401`, `404`, `500`, etc.
- Headers: response metadata.
- Body: JSON, HTML, plain text, binary data, streaming data, etc.

FastAPI decorators map HTTP methods to Python functions:

```python
from fastapi import FastAPI, status

app = FastAPI()


@app.get("/items")
async def list_items() -> list[dict[str, str]]:
    return [{"name": "notebook"}]


@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item() -> dict[str, str]:
    return {"name": "notebook"}


@app.put("/items/{item_id}")
async def replace_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


@app.patch("/items/{item_id}")
async def update_item(item_id: int) -> dict[str, int]:
    return {"item_id": item_id}


@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int) -> None:
    return None
```

Use methods intentionally:

- `GET`: read data. Should not mutate server state.
- `POST`: create a resource or trigger a command.
- `PUT`: replace a resource.
- `PATCH`: partially update a resource.
- `DELETE`: delete a resource.

Status code conventions:

- `200 OK`: successful read or update with a response body.
- `201 Created`: created a resource.
- `202 Accepted`: request accepted for later processing.
- `204 No Content`: successful request with no response body.
- `400 Bad Request`: malformed or invalid request beyond normal validation.
- `401 Unauthorized`: missing or invalid authentication.
- `403 Forbidden`: authenticated but not allowed.
- `404 Not Found`: resource does not exist or is hidden from caller.
- `409 Conflict`: state conflict, such as duplicate unique value.
- `422 Unprocessable Entity`: FastAPI validation error by default.
- `500 Internal Server Error`: server bug or unexpected failure.

Route order matters when paths can overlap:

```python
@app.get("/users/me")
async def read_me() -> dict[str, str]:
    return {"user": "current"}


@app.get("/users/{user_id}")
async def read_user(user_id: int) -> dict[str, int]:
    return {"user_id": user_id}
```

Declare `/users/me` before `/users/{user_id}` so `me` is not interpreted as a path parameter.

## Type Hints and Validation

FastAPI uses type hints as runtime instructions. This is different from normal Python, where type hints are mostly for tooling.

```python
@app.get("/items/{item_id}")
async def read_item(item_id: int, include_details: bool = False) -> dict[str, object]:
    return {"item_id": item_id, "include_details": include_details}
```

Request:

```text
GET /items/123?include_details=true
```

FastAPI parses:

- `item_id` from the path and converts it to `int`.
- `include_details` from the query string and converts it to `bool`.

If the client sends `/items/not-an-int`, FastAPI rejects it with a validation response before your function runs.

### Annotated

Prefer `typing.Annotated` for attaching FastAPI validation metadata to types:

```python
from typing import Annotated

from fastapi import FastAPI, Path, Query

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(
    item_id: Annotated[int, Path(gt=0, description="Database ID")],
    q: Annotated[str | None, Query(min_length=2, max_length=50)] = None,
) -> dict[str, object]:
    return {"item_id": item_id, "q": q}
```

Read this as:

- The Python type is `int`.
- The value comes from the path.
- It must be greater than zero.
- Documentation should show the description.

Common metadata helpers:

- `Path`: path parameters.
- `Query`: query string parameters.
- `Body`: JSON request body values.
- `Field`: Pydantic model fields.
- `Header`: headers.
- `Cookie`: cookies.
- `Form`: form fields.
- `File`: uploaded files.
- `Depends`: dependencies.
- `Security`: security dependencies with scopes.

### Common Types

FastAPI and Pydantic handle many built-in and standard-library types:

```python
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal
from uuid import UUID

from fastapi import FastAPI

app = FastAPI()


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


@app.get("/reports/{report_id}")
async def get_report(
    report_id: UUID,
    created_after: datetime | None = None,
    effective_on: date | None = None,
    min_total: Decimal | None = None,
    order: SortOrder = SortOrder.asc,
    status: Literal["draft", "published", "archived"] = "published",
) -> dict[str, object]:
    return {
        "report_id": report_id,
        "created_after": created_after,
        "effective_on": effective_on,
        "min_total": min_total,
        "order": order,
        "status": status,
    }
```

These types affect validation, JSON serialization, and OpenAPI documentation.

## Request Data

FastAPI decides where a parameter comes from based on its name, type, and default value.

General rules:

- Parameters declared in the path string are path parameters.
- Simple scalar parameters not in the path are query parameters by default.
- Pydantic models are request bodies by default.
- `Path`, `Query`, `Header`, `Cookie`, `Body`, `Form`, and `File` make the source explicit.

### Path Parameters

```python
from typing import Annotated

from fastapi import FastAPI, Path

app = FastAPI()


@app.get("/products/{sku}")
async def read_product(
    sku: Annotated[str, Path(pattern=r"^[A-Z0-9-]+$", min_length=3, max_length=40)]
) -> dict[str, str]:
    return {"sku": sku}
```

Path parameters are part of route identity. They should usually be required and stable.

### Query Parameters

```python
from typing import Annotated, Literal

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/products")
async def list_products(
    q: Annotated[str | None, Query(min_length=2, max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    sort: Literal["name", "created_at", "price"] = "created_at",
) -> dict[str, object]:
    return {"q": q, "limit": limit, "offset": offset, "sort": sort}
```

Query parameters are ideal for filters, pagination, sorting, and flags.

For repeated query params:

```python
from typing import Annotated

from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/search")
async def search(tags: Annotated[list[str], Query()] = []) -> dict[str, list[str]]:
    return {"tags": tags}
```

Request:

```text
GET /search?tags=python&tags=api
```

### Query Parameter Models

When related query parameters appear on many routes, define a Pydantic model:

```python
from typing import Annotated, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI()


class Pagination(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"


@app.get("/events")
async def list_events(params: Annotated[Pagination, Query()]) -> Pagination:
    return params
```

`extra="forbid"` rejects unknown query parameters.

### Request Body

Use Pydantic models for JSON bodies:

```python
from pydantic import BaseModel, Field
from fastapi import FastAPI

app = FastAPI()


class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str | None = None
    price: float = Field(gt=0)
    tags: list[str] = []


@app.post("/items")
async def create_item(item: ItemCreate) -> dict[str, object]:
    return {"item": item}
```

Request:

```json
{
  "name": "Notebook",
  "description": "Hardcover",
  "price": 12.5,
  "tags": ["stationery", "paper"]
}
```

FastAPI validates the body before calling your function.

### Pydantic Models

A Pydantic model describes the shape of data:

```python
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Address(BaseModel):
    line1: str
    line2: str | None = None
    city: str
    state: str = Field(min_length=2, max_length=2)
    postal_code: str


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)
    address: Address | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str
    created_at: datetime
```

Important Pydantic v2 methods:

```python
payload = UserCreate(
    email="ada@example.com",
    password="correct horse battery staple",
    display_name="Ada",
)

data = payload.model_dump()
json_ready = payload.model_dump(mode="json")
json_text = payload.model_dump_json()
```

`ConfigDict(from_attributes=True)` allows Pydantic to read attributes from ORM objects, not just dictionaries.

### Body Metadata

Use `Field` for fields inside Pydantic models:

```python
from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=80,
        examples=["Mechanical keyboard"],
    )
    price_cents: int = Field(gt=0, examples=[12900])
```

Use `Body` for top-level body parameters:

```python
from typing import Annotated

from fastapi import Body, FastAPI

app = FastAPI()


@app.post("/commands/reindex")
async def reindex(
    dry_run: Annotated[bool, Body()],
    reason: Annotated[str, Body(min_length=10)],
) -> dict[str, object]:
    return {"dry_run": dry_run, "reason": reason}
```

The request body for this route is:

```json
{
  "dry_run": true,
  "reason": "Rebuild index after schema migration"
}
```

### Partial Updates

For `PATCH`, define an update model where fields are optional, then use `exclude_unset=True`:

```python
from pydantic import BaseModel, Field


class ItemUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = None
    price: float | None = Field(default=None, gt=0)


stored = {"name": "Notebook", "description": "Hardcover", "price": 12.5}
patch = ItemUpdate(description="Softcover")
updates = patch.model_dump(exclude_unset=True)
stored.update(updates)
```

Do not use a create model for partial updates, because required create fields should not all be required during a patch.

### Headers and Cookies

```python
from typing import Annotated

from fastapi import Cookie, FastAPI, Header

app = FastAPI()


@app.get("/context")
async def read_context(
    user_agent: Annotated[str | None, Header()] = None,
    request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict[str, str | None]:
    return {
        "user_agent": user_agent,
        "request_id": request_id,
        "session_id": session_id,
    }
```

FastAPI converts Python parameter names like `user_agent` to header names like `user-agent` by default. Use `alias` when you need exact names.

### Forms

HTML forms and OAuth2 password flows send form-encoded data, not JSON:

```python
from typing import Annotated

from fastapi import FastAPI, Form

app = FastAPI()


@app.post("/login")
async def login(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> dict[str, str]:
    return {"username": username}
```

Install support with `fastapi[standard]` or `python-multipart`.

### File Uploads

For small files, bytes are acceptable:

```python
from typing import Annotated

from fastapi import FastAPI, File

app = FastAPI()


@app.post("/upload-bytes")
async def upload_bytes(file: Annotated[bytes, File()]) -> dict[str, int]:
    return {"size": len(file)}
```

For larger files, use `UploadFile`:

```python
from typing import Annotated

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/upload")
async def upload(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
    }
```

`UploadFile` gives you filename, content type, and a file-like object that can be read asynchronously.

## Response Data

FastAPI returns JSON by default. It can serialize dicts, lists, Pydantic models, datetimes, UUIDs, and many other types.

### Return Type Annotations

```python
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()


class Item(BaseModel):
    id: int
    name: str
    price: float


@app.get("/items/{item_id}")
async def read_item(item_id: int) -> Item:
    return Item(id=item_id, name="Notebook", price=12.5)
```

The return type helps FastAPI generate docs and validate/serialize the output.

### response_model

Use `response_model` when the Python return type is not the public API shape, or when you want to filter fields:

```python
from pydantic import BaseModel, EmailStr
from fastapi import FastAPI

app = FastAPI()


class UserInDB(BaseModel):
    id: int
    email: EmailStr
    hashed_password: str


class UserPublic(BaseModel):
    id: int
    email: EmailStr


@app.get("/users/{user_id}", response_model=UserPublic)
async def read_user(user_id: int) -> UserInDB:
    return UserInDB(
        id=user_id,
        email="ada@example.com",
        hashed_password="never-return-this-directly",
    )
```

The response will not include `hashed_password`.

Use separate models:

- `UserCreate`: input for registration.
- `UserUpdate`: input for partial updates.
- `UserRead` or `UserPublic`: output to clients.
- `UserInDB`: internal persistence representation, if needed.

### Status Codes

```python
from fastapi import FastAPI, status
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    name: str


@app.post(
    "/items",
    response_model=Item,
    status_code=status.HTTP_201_CREATED,
)
async def create_item(item: Item) -> Item:
    return item
```

You can also set status dynamically with a `Response` parameter:

```python
from fastapi import FastAPI, Response, status

app = FastAPI()
items: dict[str, str] = {}


@app.put("/items/{key}")
async def upsert_item(key: str, value: str, response: Response) -> dict[str, str]:
    if key not in items:
        response.status_code = status.HTTP_201_CREATED
    items[key] = value
    return {"key": key, "value": value}
```

### Direct Responses

Return a `Response` subclass when you need control over content, headers, media type, redirects, files, or streaming:

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

app = FastAPI()


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get("/page", response_class=HTMLResponse)
async def page() -> str:
    return "<h1>Hello</h1>"


@app.get("/old")
async def old() -> RedirectResponse:
    return RedirectResponse(url="/new", status_code=307)
```

Common response classes:

- `JSONResponse`
- `ORJSONResponse`
- `HTMLResponse`
- `PlainTextResponse`
- `RedirectResponse`
- `StreamingResponse`
- `FileResponse`
- `Response`

If you return a `Response` object directly, FastAPI does not apply normal response-model filtering to its content. Use that power deliberately.

### Cookies and Headers

```python
from fastapi import FastAPI, Response

app = FastAPI()


@app.post("/sessions")
async def create_session(response: Response) -> dict[str, str]:
    response.set_cookie(
        key="session_id",
        value="abc123",
        httponly=True,
        secure=True,
        samesite="lax",
    )
    response.headers["X-Request-ID"] = "req-123"
    return {"status": "created"}
```

For auth cookies, use `httponly=True`, `secure=True`, and an appropriate `samesite` value. Do not store sensitive data in unsigned plain cookies.

## Errors

Use `HTTPException` for intentional API errors:

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()
items = {1: {"id": 1, "name": "Notebook"}}


@app.get("/items/{item_id}")
async def read_item(item_id: int) -> dict[str, object]:
    item = items.get(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
        )
    return item
```

FastAPI returns structured JSON:

```json
{
  "detail": "Item not found"
}
```

For auth failures, include `WWW-Authenticate`:

```python
from fastapi import HTTPException, status


def credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

### Custom Exception Handlers

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


class DomainError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message


@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"detail": exc.message, "path": str(request.url.path)},
    )
```

Use custom handlers when you want consistent error envelopes across the whole API.

## Dependencies

Dependency injection is one of FastAPI's most important features. A dependency is a callable that FastAPI executes to provide a value or enforce behavior.

### Function Dependencies

```python
from typing import Annotated

from fastapi import Depends, FastAPI, Query

app = FastAPI()


async def pagination(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, int]:
    return {"limit": limit, "offset": offset}


@app.get("/items")
async def list_items(
    page: Annotated[dict[str, int], Depends(pagination)]
) -> dict[str, dict[str, int]]:
    return {"page": page}
```

The dependency can have its own path, query, header, cookie, body, and dependency parameters.

### Typed Dependency Aliases

Create aliases to reduce repetition:

```python
from typing import Annotated

from fastapi import Depends

Pagination = Annotated[dict[str, int], Depends(pagination)]


@app.get("/orders")
async def list_orders(page: Pagination) -> dict[str, dict[str, int]]:
    return {"page": page}
```

### Class Dependencies

Classes are useful when the dependency result has named attributes:

```python
from typing import Annotated

from fastapi import Depends, FastAPI, Query

app = FastAPI()


class SearchParams:
    def __init__(
        self,
        q: Annotated[str | None, Query(min_length=2)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> None:
        self.q = q
        self.limit = limit


@app.get("/search")
async def search(params: Annotated[SearchParams, Depends()]) -> dict[str, object]:
    return {"q": params.q, "limit": params.limit}
```

### Sub-Dependencies

Dependencies can depend on other dependencies:

```python
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status

app = FastAPI()


async def get_api_key(
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
) -> str:
    if api_key != "secret":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return api_key


async def get_tenant(api_key: Annotated[str, Depends(get_api_key)]) -> str:
    return "tenant-for-key"


@app.get("/tenant")
async def tenant(tenant_id: Annotated[str, Depends(get_tenant)]) -> dict[str, str]:
    return {"tenant_id": tenant_id}
```

FastAPI caches dependency results per request by default. If two dependencies require the same sub-dependency, FastAPI normally runs it once for that request.

Use `use_cache=False` only when the dependency must run every time:

```python
from typing import Annotated

from fastapi import Depends


async def nonce() -> str:
    return "new-value"


FreshNonce = Annotated[str, Depends(nonce, use_cache=False)]
```

### Dependencies for Side Effects

If you only need to enforce something and do not need the returned value, put dependencies in the decorator:

```python
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException

app = FastAPI()


async def verify_internal_token(
    token: Annotated[str | None, Header(alias="X-Internal-Token")] = None
) -> None:
    if token != "expected":
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/internal/status", dependencies=[Depends(verify_internal_token)])
async def internal_status() -> dict[str, str]:
    return {"status": "ok"}
```

### Yield Dependencies

Use `yield` dependencies for resources that need cleanup:

```python
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI

app = FastAPI()


class Connection:
    async def close(self) -> None:
        pass


async def get_connection() -> AsyncIterator[Connection]:
    connection = Connection()
    try:
        yield connection
    finally:
        await connection.close()


@app.get("/work")
async def work(connection: Annotated[Connection, Depends(get_connection)]) -> dict[str, str]:
    return {"status": "done"}
```

This pattern is used for database sessions, external clients, locks, and other request-scoped resources.

### Router and Global Dependencies

Attach dependencies to a router:

```python
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_internal_token)],
)
```

Attach dependencies to the whole app:

```python
app = FastAPI(dependencies=[Depends(verify_internal_token)])
```

Use global dependencies sparingly. They affect every route, including routes that might be expected to be public.

## Security and Authentication

FastAPI provides helpers for common security schemes and includes them in the OpenAPI schema so the docs UI can authenticate requests.

Security is not one thing. Separate the concerns:

- Authentication: who is the caller?
- Authorization: what is the caller allowed to do?
- Credential storage: how are secrets stored?
- Transport security: is traffic protected by HTTPS?
- Session/token policy: how long credentials last and how they are revoked.
- Auditability: what important events are logged.

### OAuth2 Password Bearer

The docs UI understands OAuth2 bearer tokens when you declare `OAuth2PasswordBearer`:

```python
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.security import OAuth2PasswordBearer

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@app.get("/me")
async def read_me(token: Annotated[str, Depends(oauth2_scheme)]) -> dict[str, str]:
    return {"token": token}
```

This only extracts the bearer token. You still need to validate it.

### JWT Tokens

Install:

```bash
pip install pyjwt "pwdlib[argon2]"
```

Minimal JWT utilities:

```python
from datetime import datetime, timedelta, timezone

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

SECRET_KEY = "replace-with-a-long-random-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def create_access_token(subject: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {"sub": subject, "exp": expires_at}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise InvalidTokenError("Missing subject")
    return subject
```

Never hardcode production secrets. Load them from environment variables or a secret manager.

### Login Endpoint

`OAuth2PasswordRequestForm` reads form fields named `username` and `password`:

```python
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


class Token(BaseModel):
    access_token: str
    token_type: str


fake_users = {
    "ada@example.com": {
        "email": "ada@example.com",
        "hashed_password": hash_password("correct horse battery staple"),
        "is_active": True,
    }
}


@app.post("/auth/token", response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = fake_users.get(form.username)
    if user is None or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=user["email"])
    return Token(access_token=token, token_type="bearer")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, object]:
    try:
        email = decode_access_token(token)
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = fake_users.get(email)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if not user["is_active"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


@app.get("/users/me")
async def read_me(
    current_user: Annotated[dict[str, object], Depends(get_current_user)]
) -> dict[str, object]:
    return current_user
```

Production auth notes:

- Use HTTPS everywhere.
- Hash passwords with Argon2id or bcrypt, never plaintext.
- Store only password hashes.
- Use a long random secret for HS256, or asymmetric keys for RS256/ES256 when appropriate.
- Use short-lived access tokens.
- Consider refresh-token rotation for browser or mobile sessions.
- Add revocation if users must be able to invalidate tokens before expiry.
- Do not put sensitive data in JWT payloads. JWTs are signed, not encrypted.
- Use scopes or roles for authorization.

### Scopes and Authorization

Scopes describe permissions:

```python
from typing import Annotated

from fastapi import Depends, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token",
    scopes={
        "tasks:read": "Read tasks",
        "tasks:write": "Create and update tasks",
    },
)


async def get_current_user_with_scopes(
    security_scopes: SecurityScopes,
    token: Annotated[str, Depends(oauth2_scheme)],
) -> dict[str, object]:
    user = await get_current_user(token)
    user_scopes = set(user.get("scopes", []))
    required = set(security_scopes.scopes)
    if not required.issubset(user_scopes):
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return user


@app.post("/tasks", dependencies=[Security(get_current_user_with_scopes, scopes=["tasks:write"])])
async def create_task() -> dict[str, str]:
    return {"status": "created"}
```

In real code, avoid storing permissions in untrusted request data. Load them from your database or include them in a signed token that you verify.

## Middleware, CORS, and Cross-Cutting Behavior

Middleware wraps every request and response. Use it for cross-cutting concerns such as request IDs, timing, logging, security headers, and CORS.

### Custom Middleware

```python
import time
from uuid import uuid4

from fastapi import FastAPI, Request

app = FastAPI()


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
```

Keep middleware small. If it blocks, it slows every request.

### CORS

CORS controls which browser origins can call your API from frontend JavaScript:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.example.com",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

Do not use `allow_origins=["*"]` with credentials. Be explicit in production.

### Security Headers

For APIs, useful headers often include:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`, if you do not need framing
- `Strict-Transport-Security`, when HTTPS is correctly configured

Example:

```python
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
```

If you serve browsers heavily, consider a dedicated security middleware or reverse-proxy configuration.

## Settings and Lifespan

### Environment Settings

Install:

```bash
pip install pydantic-settings
```

Define settings:

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FastAPI Demo"
    environment: str = "development"
    database_url: str = "sqlite+aiosqlite:///./app.db"
    secret_key: str = Field(min_length=32)
    access_token_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Use `@lru_cache` so settings are parsed once per process.

Inject settings:

```python
from typing import Annotated

from fastapi import Depends, FastAPI

app = FastAPI()
SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.get("/config")
async def config(settings: SettingsDep) -> dict[str, str]:
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
    }
```

Never return secrets from a real endpoint.

### Lifespan

Use lifespan for startup and shutdown logic:

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.cache = {}
    print("startup")
    yield
    print("shutdown")
    app.state.cache.clear()


app = FastAPI(lifespan=lifespan)
```

Use lifespan for:

- Creating shared clients.
- Loading ML models.
- Initializing connection pools.
- Warming caches.
- Cleaning resources on shutdown.

Avoid doing slow, per-request work at startup if it can be lazy-loaded. Avoid creating database tables in production startup; use migrations.

## Databases

FastAPI is database-agnostic. You can use PostgreSQL, SQLite, MySQL, MongoDB, Redis, Elasticsearch, or anything else.

For relational databases, common choices are:

- SQLAlchemy: the mature Python ORM and SQL toolkit.
- SQLModel: a library combining SQLAlchemy and Pydantic-style models.
- Databases/encode or async SQLAlchemy Core for SQL-first async code.

### Database Sessions as Dependencies

The standard pattern is a yield dependency:

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = "postgresql+asyncpg://app:app@localhost:5432/app"

engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

Then inject it:

```python
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

SessionDep = Annotated[AsyncSession, Depends(get_session)]
```

### SQLAlchemy Model

```python
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
```

### CRUD Route

```python
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

router = APIRouter(prefix="/notes", tags=["notes"])


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1)


class NoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    body: str


@router.post("", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def create_note(payload: NoteCreate, session: SessionDep) -> Note:
    note = Note(title=payload.title, body=payload.body)
    session.add(note)
    await session.commit()
    await session.refresh(note)
    return note


@router.get("", response_model=list[NoteRead])
async def list_notes(session: SessionDep) -> list[Note]:
    result = await session.execute(select(Note).order_by(Note.created_at.desc()))
    return list(result.scalars())


@router.get("/{note_id}", response_model=NoteRead)
async def read_note(note_id: int, session: SessionDep) -> Note:
    note = await session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note
```

### Transactions

Use one transaction per request or per explicit command:

```python
async def create_two_records(session: AsyncSession) -> None:
    async with session.begin():
        session.add(Note(title="A", body="one"))
        session.add(Note(title="B", body="two"))
```

If you call `commit()` in helper functions, you make composition harder. Prefer to commit at the service or route boundary.

### Migrations

Use Alembic for schema changes:

```bash
pip install alembic
alembic init migrations
alembic revision --autogenerate -m "create notes"
alembic upgrade head
```

Do not rely on `Base.metadata.create_all()` in production. It is fine for a tutorial or throwaway prototype, but migrations are how you manage real schema changes.

## Bigger Applications

As an app grows, split it into packages:

```text
app/
  __init__.py
  main.py
  core/
    __init__.py
    config.py
    security.py
  db.py
  models.py
  schemas.py
  deps.py
  routers/
    __init__.py
    auth.py
    users.py
    tasks.py
tests/
  test_health.py
  test_tasks.py
pyproject.toml
```

Keep responsibilities clear:

- `main.py`: create app, include routers, configure middleware.
- `core/config.py`: settings.
- `core/security.py`: hashing, tokens, security helpers.
- `db.py`: engine/session dependencies.
- `models.py`: ORM models.
- `schemas.py`: request/response models.
- `deps.py`: reusable dependencies.
- `routers/*.py`: HTTP route modules.
- `tests`: test suite.

### APIRouter

```python
from fastapi import APIRouter

router = APIRouter(prefix="/items", tags=["items"])


@router.get("")
async def list_items() -> list[dict[str, str]]:
    return [{"name": "notebook"}]
```

Include routers in `main.py`:

```python
from fastapi import FastAPI

from app.routers import items, users

app = FastAPI(title="Demo API")
app.include_router(items.router)
app.include_router(users.router)
```

Routers can have prefixes, tags, dependencies, default responses, and custom route classes. This is the main structure tool for larger FastAPI apps.

## Complete Application Walkthrough

This section builds a compact but realistic task API:

- Users register with email and password.
- Users log in and receive bearer JWTs.
- Authenticated users create, list, update, and delete their own tasks.
- SQLAlchemy async talks to PostgreSQL.
- Pydantic models define request and response contracts.
- Dependencies provide database sessions and current user.
- Tests override dependencies.
- Docker runs the app.

### pyproject.toml

```toml
[project]
name = "task-api"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fastapi[standard]",
  "sqlalchemy>=2.0",
  "asyncpg",
  "alembic",
  "pydantic-settings",
  "email-validator",
  "pyjwt",
  "pwdlib[argon2]",
]

[project.optional-dependencies]
dev = [
  "httpx",
  "pytest",
  "anyio",
  "aiosqlite",
]

[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["app*"]

[tool.fastapi]
entrypoint = "app.main:app"
```

### app/core/config.py

```python
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Task API"
    environment: str = "development"
    database_url: str = "postgresql+asyncpg://task:task@localhost:5432/task"
    secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

### app/db.py

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

### app/models.py

```python
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    tasks: Mapped[list["Task"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    done: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    owner: Mapped[User] = relationship(back_populates="tasks")
```

### app/schemas.py

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    is_active: bool
    created_at: datetime


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5000)
    done: bool | None = None


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str | None
    done: bool
    created_at: datetime
```

### app/core/security.py

```python
from datetime import datetime, timedelta, timezone

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings

password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def create_access_token(subject: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_minutes
    )
    payload = {"sub": subject, "exp": expires_at}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> str:
    payload = jwt.decode(
        token,
        settings.secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    subject = payload.get("sub")
    if not isinstance(subject, str):
        raise InvalidTokenError("Missing token subject")
    return subject
```

### app/deps.py

```python
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db import get_session
from app.models import User

SessionDep = Annotated[AsyncSession, Depends(get_session)]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
TokenDep = Annotated[str, Depends(oauth2_scheme)]


async def get_current_user(token: TokenDep, session: SessionDep) -> User:
    try:
        email = decode_access_token(token)
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await session.scalar(select(User).where(User.email == email))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
```

### app/routers/auth.py

```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.core.security import create_access_token, hash_password, verify_password
from app.deps import CurrentUser, SessionDep
from app.models import User
from app.schemas import Token, UserCreate, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, session: SessionDep) -> User:
    existing = await session.scalar(select(User).where(User.email == payload.email))
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@router.post("/token", response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
) -> Token:
    user = await session.scalar(select(User).where(User.email == form.username))
    if user is None or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Token(access_token=create_access_token(subject=user.email))


@router.get("/me", response_model=UserRead)
async def read_me(current_user: CurrentUser) -> User:
    return current_user
```

### app/routers/tasks.py

```python
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from app.deps import CurrentUser, SessionDep
from app.models import Task
from app.schemas import TaskCreate, TaskRead, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["tasks"])


async def get_owned_task(task_id: int, session: SessionDep, current_user: CurrentUser) -> Task:
    task = await session.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == current_user.id)
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


OwnedTask = Annotated[Task, Depends(get_owned_task)]


@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
async def create_task(
    payload: TaskCreate,
    session: SessionDep,
    current_user: CurrentUser,
) -> Task:
    task = Task(
        owner_id=current_user.id,
        title=payload.title,
        description=payload.description,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


@router.get("", response_model=list[TaskRead])
async def list_tasks(
    session: SessionDep,
    current_user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Task]:
    result = await session.execute(
        select(Task)
        .where(Task.owner_id == current_user.id)
        .order_by(Task.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(result.scalars())


@router.get("/{task_id}", response_model=TaskRead)
async def read_task(task: OwnedTask) -> Task:
    return task


@router.patch("/{task_id}", response_model=TaskRead)
async def update_task(
    payload: TaskUpdate,
    task: OwnedTask,
    session: SessionDep,
) -> Task:
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(task, key, value)

    await session.commit()
    await session.refresh(task)
    return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task: OwnedTask, session: SessionDep) -> None:
    await session.delete(task)
    await session.commit()
```

The `OwnedTask` dependency centralizes authorization: every route that accepts it can only see tasks owned by the current user.

### app/main.py

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import auth, tasks


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(auth.router)
    app.include_router(tasks.router)
    return app


app = create_app()
```

### Running the App

Development:

```bash
SECRET_KEY="$(openssl rand -hex 32)" fastapi dev app/main.py
```

Production-ish local run:

```bash
SECRET_KEY="$(openssl rand -hex 32)" fastapi run app/main.py
```

Database setup with Alembic:

```bash
alembic init migrations
alembic revision --autogenerate -m "create users and tasks"
alembic upgrade head
```

### Dockerfile

```Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY migrations ./migrations
COPY alembic.ini ./

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  api:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://task:task@db:5432/task
      SECRET_KEY: change-this-in-real-deployments
    ports:
      - "8000:8000"
    depends_on:
      - db

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: task
      POSTGRES_USER: task
      POSTGRES_PASSWORD: task
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

In real production, do not bake secrets into compose files or images. Use your platform's secret manager.

## Testing

FastAPI tests usually use `TestClient` for normal synchronous tests or `httpx.AsyncClient` for async tests.

### TestClient

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

### Dependency Overrides

Override dependencies in tests:

```python
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import get_session
from app.main import app
from app.models import Base

test_engine = create_async_engine(
    "sqlite+aiosqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


async def override_get_session() -> AsyncIterator[AsyncSession]:
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
async def database():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.dependency_overrides[get_session] = override_get_session
    yield
    app.dependency_overrides.clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.anyio
async def test_register_and_login() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        register = await client.post(
            "/auth/register",
            json={
                "email": "ada@example.com",
                "password": "correct horse battery staple",
            },
        )
        assert register.status_code == 201

        login = await client.post(
            "/auth/token",
            data={
                "username": "ada@example.com",
                "password": "correct horse battery staple",
            },
        )
        assert login.status_code == 200
        assert login.json()["token_type"] == "bearer"
```

Testing principles:

- Test public behavior, not framework internals.
- Override external services and database dependencies.
- Use a separate test database.
- Assert status codes, response shapes, and side effects.
- Test auth boundaries: unauthenticated, authenticated, wrong user, wrong scope.
- Test validation: missing fields, bad types, out-of-range values.
- Test OpenAPI generation if clients depend on it.

### Testing WebSockets

```python
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    data = await websocket.receive_text()
    await websocket.send_text(f"echo: {data}")


def test_websocket() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("hello")
        assert websocket.receive_text() == "echo: hello"
```

## Advanced Interfaces

### Request Object

Use `Request` when you need raw request details:

```python
from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/whoami")
async def whoami(request: Request) -> dict[str, str]:
    return {
        "client": request.client.host if request.client else "unknown",
        "url": str(request.url),
    }
```

Prefer typed parameters for normal data. Use `Request` for lower-level access.

### Static Files

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
```

Mounted static apps are independent sub-applications. They do not appear as normal path operations in your OpenAPI schema.

### Templates

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"title": "Dashboard"},
    )
```

FastAPI can serve HTML, but if the project is mainly server-rendered pages, compare carefully with Django, Flask, or Starlette.

### Streaming Responses

```python
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()


def rows() -> Iterator[str]:
    yield "id,name\n"
    yield "1,Ada\n"
    yield "2,Grace\n"


@app.get("/export.csv")
async def export_csv() -> StreamingResponse:
    return StreamingResponse(
        rows(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export.csv"},
    )
```

Use streaming when the result is too large or too slow to build in memory before responding.

### Background Tasks

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()


def send_email(email: str, subject: str) -> None:
    print(f"Send email to {email}: {subject}")


@app.post("/invite")
async def invite(email: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(send_email, email, "You are invited")
    return {"status": "queued"}
```

Background tasks run after the response is sent in the same process. They are good for small follow-up work. For durable jobs, retries, and heavy workloads, use a queue such as Celery, RQ, Dramatiq, Arq, or a platform-native queue.

### WebSockets

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            await websocket.send_text(f"echo: {message}")
    except WebSocketDisconnect:
        pass
```

WebSocket routes can use dependencies. Authentication usually happens with cookies, query parameters, or an initial message, because browser WebSocket APIs do not let you set arbitrary headers as easily as `fetch`.

### Mounted Sub-Applications

```python
from fastapi import FastAPI

main_app = FastAPI()
admin_app = FastAPI()


@admin_app.get("/health")
async def admin_health() -> dict[str, str]:
    return {"status": "admin-ok"}


main_app.mount("/admin", admin_app)
```

Mounted apps have their own routes and docs. This is useful for independent apps, static files, or legacy WSGI apps.

### Behind a Proxy

If a reverse proxy serves your app under a prefix, configure `root_path`:

```python
app = FastAPI(root_path="/api/v1")
```

When deploying behind proxies, ensure forwarded headers are handled by your server or platform. This affects URL generation, HTTPS detection, client IPs, and redirects.

## Deployment

Deployment means making the app reliably available to users. A production deployment needs more than "run this command".

Core deployment concerns:

- HTTPS termination.
- Process startup on boot.
- Process restart on failure.
- Worker/process count.
- Memory limits.
- Health checks.
- Logging.
- Metrics and tracing.
- Database migrations.
- Secrets.
- Static assets, if any.
- Reverse proxy and root path behavior.

### Server Commands

Single process:

```bash
fastapi run app/main.py --host 0.0.0.0 --port 8000
```

Multiple worker processes:

```bash
fastapi run app/main.py --host 0.0.0.0 --port 8000 --workers 4
```

Or with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Worker count depends on CPU, memory, workload, and deployment platform. More workers can increase throughput but also duplicate memory and startup work.

In containers and Kubernetes, prefer one Uvicorn process per container in many cases, then scale replicas at the orchestrator level. Use multiple workers inside a container only when that matches your platform and resource model.

### Docker Deployment Checklist

- Use a slim official Python base image or a hardened internal base.
- Install only runtime dependencies in the final image.
- Do not copy local `.env` files into the image.
- Run migrations as a release step, not inside every app worker.
- Expose a health endpoint.
- Send logs to stdout/stderr.
- Set memory and CPU limits.
- Use platform secrets for `SECRET_KEY`, database URLs, and API keys.

### Health Checks

```python
from sqlalchemy import text


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", include_in_schema=False)
async def ready(session: SessionDep) -> dict[str, str]:
    await session.execute(text("select 1"))
    return {"status": "ready"}
```

Use `/health` for liveness and `/ready` for dependency readiness. Keep liveness cheap.

### Observability

At minimum, production APIs need:

- Structured logs with request IDs.
- Error tracking.
- Metrics: request count, latency, error rate.
- Dependency metrics: database latency, queue depth, external API latency.
- Traces for complex distributed systems.

OpenTelemetry is the usual standard for traces and metrics.

## Best Practices

### API Design

- Use nouns for resources: `/users`, `/tasks`, `/orders`.
- Use HTTP methods for actions on resources.
- Use command-style endpoints when there is no natural resource: `/reports/{id}/publish`.
- Use consistent pluralization.
- Version deliberately if clients require stability: `/v1`.
- Use pagination for list endpoints.
- Use filtering and sorting through query parameters.
- Use `201` for creation and include the created representation.
- Use `204` for successful deletes with no body.
- Keep response envelopes consistent if you choose to use envelopes.

### Models

- Keep input and output models separate.
- Never return password hashes or internal secrets.
- Prefer `response_model` for public API boundaries.
- Use `Field`, `Query`, and `Path` constraints close to the boundary.
- Use `ConfigDict(from_attributes=True)` for ORM output models.
- For patches, use optional fields plus `exclude_unset=True`.
- Avoid leaking database models directly into request schemas.

### Dependencies

- Use dependencies for auth, DB sessions, settings, clients, pagination, and repeated filters.
- Keep dependency graphs understandable.
- Centralize authorization checks that are reused.
- Use `yield` dependencies for resources that need cleanup.
- Override dependencies in tests instead of monkeypatching route internals.

### Async and Sync

- Use `async def` when awaiting async libraries.
- Use normal `def` for CPU-light synchronous work or blocking libraries FastAPI can run in a threadpool.
- Do not call blocking I/O inside `async def`; it blocks the event loop.
- For CPU-heavy work, use a worker process, task queue, or specialized service.
- Use async database drivers if your route code is async and database I/O is frequent.

### Database

- Use migrations.
- Keep transactions explicit.
- Avoid N+1 queries.
- Add indexes for common filters and joins.
- Validate ownership and tenant constraints in queries, not only after fetching.
- Do not commit inside low-level repository helpers unless that is the explicit boundary.
- Use connection pool settings appropriate for worker count and database limits.

### Security

- Enforce HTTPS in production.
- Store secrets outside the repo.
- Hash passwords with Argon2id or bcrypt.
- Use short-lived access tokens.
- Treat JWT payloads as readable by users.
- Validate issuer, audience, subject, expiry, and algorithm when using external identity providers.
- Restrict CORS origins.
- Do not put secrets in logs.
- Rate-limit login and sensitive endpoints.
- Use least-privilege database users.

### Documentation

- Use tags to group routes.
- Use summaries and descriptions for non-obvious endpoints.
- Hide operational endpoints with `include_in_schema=False` when appropriate.
- Keep generated OpenAPI as a contract for frontend and client SDKs.
- Consider stable `operation_id` values if generating SDKs.

### Performance

- Measure before optimizing.
- Add pagination and limits to list routes.
- Stream large responses.
- Use `ORJSONResponse` only after verifying serialization is a bottleneck.
- Avoid unnecessary response-model work for huge responses if you already validate data elsewhere.
- Use caching where it matches correctness requirements.
- Keep Pydantic models focused; deeply nested huge models cost CPU.

### Project Structure

- Keep route modules thin.
- Move business rules into service functions when they grow.
- Move persistence details into repository/query functions when they grow.
- Avoid circular imports by keeping shared dependencies in `deps.py`.
- Prefer an app factory for tests and configurable app creation.
- Keep environment-specific configuration out of code.

## Troubleshooting

### 422 Unprocessable Entity

FastAPI returns `422` when validation fails. Check:

- Did the client send JSON with `Content-Type: application/json`?
- Is a query parameter being sent in the body or vice versa?
- Does the path value match the annotated type?
- Are required fields missing?
- Are field names correct, including aliases?

### 401 vs 403

Use:

- `401` when the caller is unauthenticated or the credential is invalid.
- `403` when the caller is authenticated but lacks permission.

For bearer auth, `401` responses should usually include `WWW-Authenticate: Bearer`.

### CORS Fails in Browser but curl Works

CORS is enforced by browsers, not by curl. Check:

- Exact frontend origin: scheme, host, and port.
- `allow_origins`.
- `allow_credentials`.
- Preflight `OPTIONS` requests.
- Allowed headers and methods.
- Whether the browser needs exposed response headers.

### Blocking Async Routes

This is bad in `async def`:

```python
import requests


@app.get("/external")
async def external():
    return requests.get("https://example.com").json()
```

Use an async client:

```python
import httpx


@app.get("/external")
async def external():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com")
    return response.json()
```

Or make the route `def` if you must use blocking code and the workload is small.

### Response Model Errors

If FastAPI raises a response validation error, your route returned data that does not match the declared response model. Treat this as a server bug. Fix either the data or the response contract.

### Duplicate Database Connections

If every worker creates its own pool, total database connections are:

```text
workers * pool_size + overflow
```

Set pool sizes with your database limit in mind.

### Startup Runs More Than Once

With reloaders or multiple workers, startup code can run multiple times across processes. Do not put non-idempotent global side effects in module import or startup code.

## Quick Reference

### Core Imports

```python
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    WebSocket,
    status,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict, Field
```

### Common Decorator Options

```python
@router.get(
    "/items/{item_id}",
    response_model=ItemRead,
    status_code=status.HTTP_200_OK,
    tags=["items"],
    summary="Read one item",
    description="Returns a single item visible to the current user.",
    responses={404: {"description": "Item not found"}},
)
```

### Common Parameter Patterns

```python
item_id: Annotated[int, Path(gt=0)]
q: Annotated[str | None, Query(min_length=2, max_length=100)] = None
payload: Annotated[dict, Body()]
authorization: Annotated[str | None, Header()] = None
session_id: Annotated[str | None, Cookie()] = None
file: Annotated[UploadFile, File()]
form: Annotated[OAuth2PasswordRequestForm, Depends()]
```

### Common Response Patterns

```python
raise HTTPException(status_code=404, detail="Not found")
return JSONResponse(status_code=202, content={"status": "queued"})
return RedirectResponse(url="/new-path")
return StreamingResponse(iterator, media_type="text/csv")
return FileResponse("report.pdf", media_type="application/pdf")
```

## Sources

Primary official docs consulted:

- FastAPI home and installation: https://fastapi.tiangolo.com/
- First steps and generated docs: https://fastapi.tiangolo.com/tutorial/first-steps/
- Path, query, and body parameters: https://fastapi.tiangolo.com/tutorial/path-params/ , https://fastapi.tiangolo.com/tutorial/query-params/ , https://fastapi.tiangolo.com/tutorial/body/
- Query parameter models: https://fastapi.tiangolo.com/tutorial/query-param-models/
- Response models and status codes: https://fastapi.tiangolo.com/tutorial/response-model/ , https://fastapi.tiangolo.com/tutorial/status-code/
- Error handling: https://fastapi.tiangolo.com/tutorial/handling-errors/
- Dependencies: https://fastapi.tiangolo.com/tutorial/dependencies/
- Security and JWT: https://fastapi.tiangolo.com/tutorial/security/first-steps/ , https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
- Middleware and CORS: https://fastapi.tiangolo.com/tutorial/middleware/ , https://fastapi.tiangolo.com/tutorial/cors/
- SQL databases and larger apps: https://fastapi.tiangolo.com/tutorial/sql-databases/ , https://fastapi.tiangolo.com/tutorial/bigger-applications/
- Background tasks, static files, and testing: https://fastapi.tiangolo.com/tutorial/background-tasks/ , https://fastapi.tiangolo.com/tutorial/static-files/ , https://fastapi.tiangolo.com/tutorial/testing/
- Advanced responses, requests, templates, WebSockets, lifespan, settings, and async tests: https://fastapi.tiangolo.com/advanced/custom-response/ , https://fastapi.tiangolo.com/advanced/using-request-directly/ , https://fastapi.tiangolo.com/advanced/templates/ , https://fastapi.tiangolo.com/advanced/websockets/ , https://fastapi.tiangolo.com/advanced/events/ , https://fastapi.tiangolo.com/advanced/settings/ , https://fastapi.tiangolo.com/advanced/async-tests/
- FastAPI CLI and deployment: https://fastapi.tiangolo.com/fastapi-cli/ , https://fastapi.tiangolo.com/deployment/ , https://fastapi.tiangolo.com/deployment/server-workers/ , https://fastapi.tiangolo.com/deployment/docker/
