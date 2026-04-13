---
aliases:
  - SODA
  - Socrata Open Data API
---


A software platform that many government agencies use to publish and manage their open data.
When a city IT department wants to put a dataset on the internet, they often upload it to Socrata, which handles hosting, search, and API access.

Places like:
- data.alcity.org
- da.lacounty.gov
- ... and hundreds of other  government portals are *all* Socrata instances -- same software, different domains.

Every dataset on a Socrata portal automatically gets a REST API called SODA (Socrata Open Data API). 

Without an app token, Socrata rate-limits you aggressive (~1 request a second, ~1000 rows/request), but with a free app token (register), limits become generous (~50,000 rows per request).
