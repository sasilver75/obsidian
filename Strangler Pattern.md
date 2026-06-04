
A way to decompose a [[Monolith]] into services without doing a big rewrite.

==You put a routing layer in front of the old system, and gradually move specific capabilities into new services; traffic for migrated behavior goes to the new service, while everything else goes to the monolith.==

Example:
- An existing commerce monolith owns orders, catalog, payments, and shipping.
- You might first extract search or product catalog.
- The frontend/API gateway routes catalog reads to the new catalog service, while order checkout still goes to the monolith.
- Over time, more routes move out until the old code path is unused, and can be deleted.

The hard part is usually data ownership; early-on, the new service may read from the monolith DB via CDC, views, APIs, or duplicated projections. Eventually, the extracted service should own its own data and expose behavior through APIs, events.

==Move ownership one capabilities at a time, with routing and data transition as the control points.==


