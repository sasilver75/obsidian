The act of making a deployed capability/change/version available to its intended users.

A release is about **exposure and access:** Who can use the new behavior, when they can use it, and under what controls.

A release may happen all at once, gradually, or selectively through mechanisms like feature flags, cohorts, regions, plans, tenants, beta programs, or traffic routing.

It's related to, but distinct from [[Deployment]]
- Deployment: The system is running the new version (e.g. it's been deployed to production)
- Release: Users are allowed to experience the new behavior


Common release strategies:
- Big-bang release: Enable the new behavior for everyone at once
- [[Feature-Flagged Release]]: Deploy code first, then turn behavior on with a flag
- [[Canary Release|Canary Release]]: Expose a small percentage or cohort of users first, then expand
- Ring release: Release through ordered groups, such as employees -> beta users -> one region -> all users
- Phased rollout: Gradually increase exposure by percentage, region, platform, tenant, or plan
- Dark launch: Run new behavior invisibly without presenting it to users, often to test load or correctness
- Shadow release: Send copied production traffic to the new path but discard its output
- [[AB Testing|AB Test]]: Release variants to different groups to measure product impact
- Beta/ preview release:e Make the feature availability to opted-in users before general availability
- Internal/dogfood release: Release to the company or team first.
- Limited availability: Release only to selected customers, accounts, plans, markets, or partners.



