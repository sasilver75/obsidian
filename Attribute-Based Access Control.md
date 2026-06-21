---
aliases:
  - ABAC
---
An [[Authorization]] model where access decisions are made by evaluating attributes about the requester, resource, action, and environment against a policy.
- A common tool for implementation is [[Open Policy Agent]] (OPA)

This is different from simpler models like [[Role-Based Access Control]] where access is granted mainly because a user has a role.

Many authorization decisions depend on *more than identity or role!*
- A doctor may view a patient record only if the doctor is assigned to that patient.
- An employee may download a report only during business hours.
- A support agent may view customer data only for customers in the agent's region.
- A contractor may access a system only from a managed device.
- A user may approve an expense only if the amount is below the user's approval limit.

Role-based access control can express some of this, but often becomes awkward when you start creating roles like `SeniorFinanceMAnager_US_West_Under50k`

==ABAC’s main advantage is expressiveness==. It can model policies that depend on real-world context.
- The cost is ==complexity==. ABAC systems need reliable attributes, clear policy semantics, careful testing, and good audit tooling. A policy might look reasonable in isolation but behave unexpectedly when ==*combined*== with other policies.
	- A poorly designed ABAC policy (or set of policies) can be less safe than a simple role-based policy.

ABAC answers by collecting attributes:
1. Subject attributes: Alice's department, clearance level, employment type, location.
2. Resource attributes: Document classification, owner department, region, project, sensitivity.
3. Action attributes: Read/write/delete/approve/export
4. Environment attributes: Current time, IP address, device trust level, network, risk score

A typical ABAC system has these pieces:
1. Subject: The actor making the request, usually a user, service account, workload, or process.
2. Resource: The object being accessed, such as a file, database row, API endpoint, invoice, or record.
3. Action: The operation being requested, such as read, edit, delete, transfer, approve.
4. Attributes: Facts about the subject, resource, action, or context.
5. Policy: A rule or set of rules that says when access should be allowed or denied.
6. Policy Decision Point: The component that evaluates the policy.
7. Policy Enforcement Point: The component that blocks or permits the actual request.

> Allow access if the user’s department matches the document’s department, the document is not classified above the user’s clearance level, and the request comes from a trusted device.

> Employees may view salary records only if they are in Human Resources, are located in the same country as the employee whose record they are viewing, and are using a managed device.

