---
aliases:
  - AuthKit
  - WorkOS AuthKit
---
A developer platform for making a [[Business-to-Business|B2B]] SaaS app "enterprise ready." WorkOS bundles APIs and hosted flows for [[Authentication]], [[Single Sign-On|SSO]], directory sync, admin onboarding, audit logs, [[Authorization]], fraud protection, and related enterprise IT requirements.

Mental model: WorkOS is not the enterprise customer's identity provider. The customer may still use Okta, Microsoft Entra ID, Google Workspace, OneLogin, etc. WorkOS sits between those systems and your app, normalizing the messy parts of enterprise identity and compliance behind a smaller set of APIs.

# Core Product Areas

### AuthKit
- User management and hosted login for email/password, social login, [[Single Sign-On|SSO]], magic codes, MFA, passkeys, and organization-aware auth.
- Can provide hosted UI, sessions, users, organizations, email verification, identity linking, domain verification, organization policies, roles/permissions, and [[JSON Web Token|JWT]] templates.
- Good when you want WorkOS to be the main auth layer, not just an enterprise SSO add-on.

### Enterprise SSO
- Lets enterprise users sign in through their organization's identity provider.
- Supports [[Security Assertion Markdown Language|SAML]] and [[OpenID Connect|OIDC]] providers through one app-side integration.
- Can be used as a standalone API on top of an existing auth stack, or as part of AuthKit.
- Important security detail: validate the returned WorkOS `organization_id`, not just the email domain, because enterprise organizations can include contractors, guest users, and non-corporate addresses.

### Directory Sync
- User lifecycle management through SCIM and HRIS/directory integrations.
- Lets customer IT teams provision, update, group, and deprovision users in their source-of-truth directory; WorkOS normalizes those changes and sends them to your app through events/webhooks or API reads.
- This becomes important when enterprise buyers expect automatic offboarding, group-based access, and role mapping.

### Admin Portal
- Hosted UI for customer IT admins to verify domains, configure SSO, configure Directory Sync, and set up log streams.
- Reduces the support burden of enterprise onboarding: instead of hand-walking each customer through Okta/Entra/Google setup, your app can generate a setup link and let the IT contact complete configuration in WorkOS.

### Audit Logs
- API for ingesting security and compliance events from your application.
- Events are structured around action, actor, targets, context, and timestamp.
- Useful for enterprise customers that need an inspectable/exportable paper trail of sensitive actions, e.g. sign-ins, permission changes, exports, administrative actions, or data access.

### Authorization
- [[Role-Based Access Control]] supports organization-level roles and permissions.
- Fine-Grained Authorization extends RBAC to resource-scoped and hierarchical permissions, e.g. organization -> workspace -> project -> app.
- Practical split: use embedded [[JSON Web Token|JWT]] claims for fast organization-wide checks, and the WorkOS Authorization API for resource-level checks.

### Radar
- AuthKit add-on for bot, fraud, and abuse protection during sign-in/sign-up.
- Uses behavioral signals and device fingerprinting to block, challenge, or notify on suspicious attempts.
- Built-in detections include bots, brute force, impossible travel, repeat sign-up, stale accounts, unrecognized devices, suspicious domains, disposable email domains, and sanctioned-country controls.

### Connect and MCP
- WorkOS Connect lets other applications access your users and their resources using [[OAuth]] 2.0 / [[OpenID Connect|OIDC]] patterns.
- Supports OAuth applications for user-delegated access and machine-to-machine applications for service credentials tied to an organization.
- AuthKit can also act as an [[OAuth]] authorization server for MCP servers, so AI/MCP clients can authenticate through standard metadata and token flows.

### Vault
- Encrypted key-value storage and enterprise key management for sensitive data such as tokens, passwords, certificates, customer secrets, or PII.
- Supports envelope encryption, organization/user/application-scoped key context, and BYOK integrations with cloud KMS products.

# Integration Shape

- Create a WorkOS organization per enterprise customer and store the WorkOS IDs alongside your own tenant/account records.
- Redirect users through AuthKit or the standalone SSO flow, then map the returned WorkOS user/profile back to your internal user.
- For Directory Sync, maintain a mapping from WorkOS directory users/groups to your own users/groups/roles and keep it fresh via WorkOS events/webhooks.
- Use Admin Portal links for customer self-service setup.
- Emit audit events from your own app at the point important actions happen.
- SDKs exist for Node.js, Go, Ruby, Rust, Python, PHP/Laravel, Java/Kotlin, .NET, plus AuthKit-specific SDKs for JavaScript, React, Next.js, Remix, React Router, and TanStack Start.

# When I Would Use It

- B2B SaaS moving upmarket and repeatedly blocked by SAML SSO, SCIM, audit logs, role mapping, or IT-admin onboarding requirements.
- A team that wants enterprise auth without building and maintaining dozens of IdP-specific setup paths.
- A product that already has an auth system but needs to add enterprise SSO/SCIM/audit logs as standalone modules.
- A product that wants user management, enterprise SSO, organization auth policies, and bot/fraud controls from one vendor.

Less compelling if the app is consumer-only, requires self-hosted auth infrastructure, needs unusually custom IAM semantics, or cannot absorb per-enterprise-connection pricing.

# Comparisons

- [[Auth0]]: broader identity platform and CIAM option. WorkOS feels more specifically aimed at B2B SaaS "enterprise readiness": SSO, SCIM, audit logs, admin setup, org policies, and authorization.
- Okta / Microsoft Entra ID / Google Workspace: often the customer's identity provider, not the same role as WorkOS. WorkOS helps your app integrate with them.
- [[Supabase]], Firebase Auth, or Clerk: strong general app auth choices; WorkOS matters more when enterprise procurement asks for SAML, SCIM, audit exports, domain verification, and IT-admin setup.
- OpenFGA / SpiceDB / Oso: deeper standalone authorization-system comparisons. WorkOS FGA is attractive when you want authorization tightly integrated with WorkOS identity, organizations, and roles.

# Key Terms

- Organization: the WorkOS representation of a customer tenant.
- Connection: an SSO or Directory Sync integration for a group of users, commonly one enterprise customer.
- IT contact: the customer's admin who configures SSO, Directory Sync, domain verification, or log streaming.
- Directory: the customer's source of truth for users and groups.
- Actor/action/target/context: the basic shape of an audit log event.