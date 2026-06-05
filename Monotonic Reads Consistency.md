A [[Consistency]] model.

Once a client has seen a version of data, future reads by that client will not return *older* versions.


# How is it achieved?
- The client/session records the highest version/timestamp/log position it has ***==observed==***.
- Every later read must go to a replica whose state is at least that fresh. 
- If the chosen replica is behind, the system either:
	- waits until that replica is up-to-date
	- chooses a fresher replica
	- forwards the read to the leader

# Tradeoffs
- This prevents confusing "time travel" experiences for one client, but doesn't guarantee that they get the *latest* value. 
- Requires maintaining freshness metadata, and may limit replica choice.

# Use Cases
- User timelines, support dashboards, replicated document views.
- Once a user has seen a newer state, they shouldn't refresh and then confusingly see an older one. 
- If a support agent sees "ticket assigned," a later refresh should not show it as "unassigned."

