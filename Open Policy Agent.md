---
aliases:
  - OPA
---
An open-source policy engine often used for implementing [[Attribute-Based Access Control]] (ABAC)
- Though you can also technically implement [[Role-Based Access Control|RBAC]] using OPA; it just evaluates policy logic over input data, whether that input data is the typical attribute-mode data or if it's role/permission-related data.

You write policies in a language called ==Rego==, and send in structured input (e.g. user, action, resource, request path,, token claims, environment data), and OPA returns a decision.

OPA is especially common for APIs, microservices, [[Kubernetes|K8s]] admission control, IaC checks, and sidecar-style policy enforcement. 
- OPA itself is usually your ==Policy Decision Point==, whereas your application/API gateway/Kubernetes admission controller/sider acts as the ==Policy Enforcement Point== that actually enforces the result of the policy decision.

OPA is useful when the application's authorization rules are:
- Complex
- Shared across multiple services
- Expected to change over time
- Easier to test as policy code than as scattered application logic
- Based on attributes, relationships, request metadata, or environment data


An application can send OPA facts like:
```json
{
  "user": "alice",
  "action": "read",
  "resource": {
    "type": "salary_record",
    "owner": "alice",
    "department": "finance"
  }
}
```
Example policy (LM output, unsure if "real"):
```rego
package authz

default allow := false

allow if {
  input.action == "read"
  input.resource.type == "salary_record"
  input.user == input.resource.owner
}
```
This says: "Allow a user to read a salary record iff the user owns that salary record."



# Using OPA for RBAC

You give OPA some data like...
```json
{
  "user_roles": {
    "alice": ["finance_reader"],
    "bob": ["finance_admin"]
  },
  "role_permissions": {
    "finance_reader": [
      {"action": "read", "resource_type": "invoice"}
    ],
    "finance_admin": [
      {"action": "read", "resource_type": "invoice"},
      {"action": "approve", "resource_type": "invoice"}
    ]
  }
}
```

Then, a Rego policy like this checks whether the request has a role that grants the requested permission:
```rego
package authz

default allow := false

allow if {
  role := data.user_roles[input.user][_]
  permission := data.role_permissions[role][_]

  permission.action == input.action
  permission.resource_type == input.resource.type
}
```

So that when an input is:
```json
{
  "user": "alice",
  "action": "read",
  "resource": {
    "type": "invoice",
    "id": "invoice-123"
  }
}
```
OPA can return `allow = True`