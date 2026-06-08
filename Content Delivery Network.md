---
aliases:
  - CDN
---



A CDN can cache a lot more than images and videos!
- Homepage HTML, like Reddit, NYTimes, e-commerce category pages
- API responses, such as `/api/posts?sort=hot`
- Product pages
- User-independent recommendation blocks
- Generated images/thumbnails/PDFs/previews
- Expensive computations where "fresh within 30 seconds" is fine
- Auth-adjacent content if the cache key safely includes user/role/tenant

