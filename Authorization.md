---
aliases:
  - Authz
  - Authorize
---
The step that happens after [[Authorization]].
- Once the system knows who a user is, the next step is Authorization, deciding what they can do.

It needs to check what resources or actions the user has permissions to access, and what are the denied actions for that user.

There are three main authorization models:
- [[Role-Based Access Control]] (RBAC): Assigns roles (admin, editor) to users, which have permissions attached. Most common approach.
- [[Attribute-Based Access Control]] (ABAC): More flexible/complex; Decision based on user/resource/environment attributes.
- [[Access Control List]] (ACL): Each resource has its own permissions list. Common way of managing things like Google Docs.
And we often use technologies like [[OAuth|OAuth 2.0]] and [[JSON Web Token|JWT]]s to make this work in practice.
Real systems often combine multiple models together.



# [[Role-Based Access Control]] (RBAC)
- 

# [[Attribute-Based Access Control]] (ABAC)
- Uses user ==attributes== and ==resource== attributes and ==environment== conditions to define access:
```
Policy Example

Allow read access if:
user.department == "HR" && time < 6PM && resource.confdientiality == "internal"
```
- Can be combined with RBAC.
- More flexible than RBAC, but more complex and requires good Policy management.

# [[Access Control List]] (ACL)
- Have a document like a `doc123.json` file, which has a permission list:
	- Alice: Read
	- Bob: Read/Write
	- Carol: No Access
- Real Example: Google Drive
	- Every file has its own sharing settings, and each document maintains individual permissions lists.
- Highly specific and also user-centric, which means it can be hard to scale them well in system with millions or users and objects unless you manage them carefully.



# [[OAuth|OAuth2]]: Delegated Authorization
- When one service accesses another on behalf of a user.
- Say that you want to let a third-party app read your Github repository (e.g. you're deploying an app to [[Vercel]]).
	- Instead of giving Vercel your username and password, instead Github gives them a token that represents the permissions that you approved to use. 

# Token-Based Authorization using [[JSON Web Token|JWT]] or [[Bearer Token]]s
- For authentication, most systems use a Token (typically a JWT or Bearer Token) carrying information like user id, roles like admin/editor, and scopes: erad posts, write posts, expires: 24h, ISsuer: auth.example.com)
- Token claims both identity and claims.
- Tokens are just the *mechanism,* while the authorization model (RBAC/ABAC/ACL) is what makes the actual decisions in your backend.







