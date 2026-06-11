---
aliases:
  - Bearer
---


A Bearer token just means that whoever has this token gets access. It's a pattern, not a specific method.

```
Authorization: Bearer eyJ...
```

The most common type of Bearer Token is a [[JSON Web Token]] (JWT), a signed JSON object that contains (e.g.) the user_id and email, expiration time, and other claims (roles, permissions, etc.). One the validation server we validate credentials.