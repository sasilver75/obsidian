---
aliases:
  - BFF
---
An architecture pattern where each client experience gets its own backend/API layer tailored to that frontend's needs.
- Each BFF talks to internal service, composes ata, applies client-specific ogic, and returns response shaped for that frontend.

Instead of one generic backend API serving every client (web, mobile, admin dashboard), you instead create separate backend surfaces:
- Web BFF
- Mobile BFF
- Admin BFF

Why use it:
- Helps avoid overfetching or underfetching
- Hides internal service complexity from clients
- Lets us tailor payloads to mobile vs web needs


