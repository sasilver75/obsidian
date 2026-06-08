One deployable application. A traditional monolith puts most of the application in one process and usually one codebase. This can be great early on: simple deployment, simple debugging, no network calls between internal features, easy transactions, easy local development.

The failure more is that boundaries eventually get blurry: Everything imports everything else, the database becomes a shared junk drawer, and a change in the billing logic accidentally breaks onboarding. This problem is not that it's a monolith, it's that it has become an *unstructured* monolith. See [[Modular Monolith]]!


