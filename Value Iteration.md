
![[Pasted image 20240628232605.png]]
Above: Note H="Horizon," meaning how many steps you have left before your agent basically times out. This is notation that's present in Pieter Abbeel's course, but wasn't in David Silver's.
![[Pasted image 20240628232744.png]]
 If you run this for long enough, it will converge; we'll get a stationary optimal policy.
 ![[Pasted image 20240628233314.png]]
Once we have v*, we can extract the optimal action using the Bellman equation (note)
![[Pasted image 20240628233848.png]]
The series is a geometric series, so it can be simplified as shown.

![[Pasted image 20240628235834.png]]
 k+1 here is some value of H it's basically just "at time remaining k+1" and you can see at the next step we use "k".
 ![[Pasted image 20240629000058.png]]
Once you have your optimal Q, you can easily read off your optimal Pi by just taking the action any state that maximizes the Q, among available actions.



![[Pasted image 20240625232849.png]]