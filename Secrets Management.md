The discipline of storing, distributing, rotating, auditing, and revoking sensitive values used by software.

A "secret" is usually something like:
- Database password
- API key
- [[OAuth]] client secret
- [[JSON Web Token|JWT]] signing key
- [[Transport Layer Security|TLS]] private key
- [[Secure Shell|SSH]] key
- Webhook signing secret
- Encryption key or key-wrapping credential

The short version: Secret management exists so that secrets are not scattered across source code, `.env` files, Slack, CI logs, shell history, laptops, and production servers with no control over who has them.

Without Secret Management, teams end up with:
- Secrets committed to Git
- Shared production passwords
- No audit trail of who accessed what
- No easy way to rotate credentials
- Secrets copied into many environments
- Long-lived credentials that never expire
- Hard incident response when something leaks

A proper secret manager gives you a controlled place to answer:
1. Who can access this?
2. What system used it?
3. When was it last read?
4. Can we rotate it?
5. Can we revoke it now?
6. Can we avoid humans seeing it at all?

Generally, at a high level:
```
App / user / CI job
          |
          | authenticates using identity
          v
  Secret manager
          |
          | checks policy
          v
  Returns secret, token, certificate, or short-lived credential
```

Provides:
- Encrypted storage: Secrets are encrypted at rest, often using KMS or HSM-backed keys
- Access control: [[Amazon Identity and Access Management|IAM]]/[[Role-Based Access Control|RBAC]]/policies decide who or what can read each secret
- Versioning: Secrets can have versions like `v1`, `v2`, `latest`
- Audit logs: Records reads, writes, updates, deletes, and rotations
- [[Key Rotation|Rotation]]: Replace old credentials with new ones
- Revocation: Invalidate credentials when no longer trusted
- Dynamic secrets: Generate temporary credentials on demand, instead of storing static ones


# Common Options
Cloud Native
- [[Amazon Secrets Manager|AWS Secrets Manager]]: Integrates with [[Amazon Identity and Access Management|AWS IAM]] and [[Amazon Key Management Service|AWS KMS]], supports scheduled and on-demand rotation.
- [[Google Cloud Secret Manager]]
- Azure Key Vault

Dedicated/Multi-Cloud
- [[HashiCorp Vault]]: Strong option for dynamic secrets, leases, revocation, encryption services, and multi-cloud on-prem setups. Vaults can generate credentials on demand and attach leases to dynamic secrets.
- Akeyless, CyberArk Conjur, Infisical, Doppler: Other managed or self-hosted platforms

Platform-native
- [[Kubernetes]] Secrets: Useful for passing secrets into pods, but by default, Kubernetes warns that Secrets are stored unencrypted in [[etcd]] unless encryption at rest is enabled. Access also depends heavily on RBAC.
- [[GitHub Actions]] Secrets, [[Vercel]] env vars, [[Netlify]] env vars, etc.: Good for platform-local deployment secrets, but usually not a full enterprise-wide secret lifecycle system.
