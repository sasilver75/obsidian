

![[Pasted image 20240521205554.png]]
![[Pasted image 20240521205559.png]]
(Above, in an [[N-Gram]] context) We estimate probabilities as we usually do, but we *steal* some probability from every observed sequence in our dataset, and distribute that somehow across all of the zero-count entries in the table.
There was a lot of research on smoothing strategies 10-20 years ago; We've pretty much stopped using N-Gram models, but variants of this kind of approach were used in SoTA language models even 10 years ago.