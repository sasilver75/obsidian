TD-Lambda gives us a spectrum between [[Monte-Carlo Learning]] and [[Temporal Difference Learning|TD-Learning]]  (TD Zero)
- By choosing Lambda, we can pick our Lambda so as to get to the sweet spot between Monte-Carlo and TD updates.

Forward View vs Backward View
- Forward View: Theoretical way of averaging over multiple n-step reviews
- Backward View: Accumulate eligibility traces, which tell us how much credit to assign to each state.


![[Pasted image 20240625165728.png|400]]

