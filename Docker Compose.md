
In [[Docker Compose]], instead of just running containers, we run multi-container *applications* defined in a `compose.yaml` file:
```yaml
services:
  app:
    build: .
    ports:
      - "3000:3000"

  database:
    image: postgres:17
    environment:
      POSTGRES_PASSWORD: example
```
This can start both the application and the database together:
```bash
docker compose up
```

