


_________
August 19, 2026
- Okay, so today I was noted that the NWIC NC3 people are using Version 5.26.8 of Neo4j. John also mentioned that he would be interested in the 5.26.12 version of Neo4j, which is the newest Ironbank version that seems to not have horrible CVEs.
- John mentioend that we've dumped the DB via a script to a self-hosted version (for MARFORPAC), and that Vince is the one to talk about this. John doesn't think that this should break anything, but it's hard to guess at -- and asked that I take point on verification.

> @Sam Silver, Something of a short fuse task here but we finally got a neo4j version out of the NIWC NC3 people for what they are using.
> Version 5.26.8
> We will need to verify that our stuff works on this version of Neo4j. We have dumped the db via script to a self hosted version (for MARFORPAC), Vince is the one to talk to about this. I do not suspect we should have any breaks but, quite frankly, that is really difficult to guess at. Can you take point on this verification? NC3 is hosting theirs on an isolated network so it's possible that the version they are using also doesnt pass our scan but this is the front-runner approach we have currently
> I would also be interested in 5.26.12 (it is the newest ironbank version that seems to not have horrible CVEs)

Re: ==5.26.8== and ==5.26.12==:
> I believe both have been scanned by ironbank (both are listed as non-compliant) but both have beneath the threshold of CVEs we have been given.



Note about Neo4j Versioning:
- After 5.26.0 (released Dec 6, 2024), the versioning changed to calendar versioning, so the next release was 2025.01.0 (In Feb 6, 2025), followed by 2025.02.0 (Feb 26, 2026), 2025.03.0 (March 27, 2025), etc.
	- Neo4j calls this CalVer (Calendar Versioning) instead of SemVer. `YYYY.MM.PATCH`
		- The MM is also called the "Monthly release line," e.g. `07`
	- Note that the MM identifies the *planned* release train, not necessarily the exact publication month. For example, `2025.01.0` was publicly released in early Feb 2025, but retained the January release-line name.
	- Note that even after the breakpoint (Beginning of 2025), there were still some patch versions released for the old versioning system, so things like `5.26.12` were released AFTER several `2025.XX` versions, but doesn't supersede them.


```
New-feature stream:
5.24 → 5.25 → 5.26 LTS → 2025.01 → 2025.02 → ... → 2026.07

LTS maintenance stream:
                    5.26.0 → 5.26.1 → ... → 5.26.8 → ... → 5.26.12 → ...


December 2024                   August 2026
     │                               │
     ├─ 5.26 LTS ── 5.26.8 ── 5.26.12 ── continued LTS patches
     │
     └─ 2025.01 ── 2025.02 ── … ── 2026.06 ── 2026.07
                                                   ↑
                                         Your Aura instances
```


![[Pasted image 20260819115547.png]]
It looks like our Aura database is running the ==Enterprise== edition. At the same time, 5.6 Sol Xhigh looked at our codebase and didn't seem to think that we were using any of the enterprise-specific features.[
- Note: 2026.07 is the monthly Aura release track deployed to the instances, while 5.2-aura is some sort of  Aura-specific kernel/compatability identifier returned to clients. It's not ordinary Neo4j 5.27.


Note about Neo4j enterprise vs community
- 


So we're running version ==2026.07== in both `Smack Neo Dev` and `Smack Neo 1` instances in Aura.
And we're asking about the earlier ==5.26.8== (June 9, 2025) and the ==5.26.12== (Sep 4, 2025) versions.

As John said there's apparently a migration script that's in 


Talking with RPC (and some Vince) at lunch:
- Yeah, there are two migration scripts.... Vince said that they're basically the same thing, but to use the Smithy one.
- RPC said that we have an enterprise license, but that Clint said that we'd never renew it ((Idk when it expires)). The Neo4j licensing/sales teams suck apparently, never figured out how to do it.
- RPC 


Okay, so it turns out that the 2026.07 release (which is what we're using on Aura anyways) is not on Ironbank, but it IS on Chainguard, and RPC things the vuln scans he did on it look promising.


Notes on the migration process:
- Note that while it imported the same number of nodes (145572), it didn't import the same number of relationships (253181 in the seeded one vs 253718 in dev, for instance).
- So this was a weird timing thing, I think?


RPC is working on something to let me run Neo4j locally in a correct way, instead of the Clanker way that I had done it.

I recorded our conversation as a video

His experience clicking around was basically just that the Athena graph loaded and that was good enough, I might want to do more, even though it passed the automated smoke checks and stuff.

_______

