
Google: "Numpy Broadcasting Rules"
( You can broadcast a rank 3 think with a rank 1 thing )

tenesor[1.,2,3] * tensor([2])
Would be us broadcasting a scalar over a vector. It's copying that two into each of htese spots and multiplying them together; it doesn't use up any memory to do that coopying; it's a virtual copying, if you like.

