
I take the queries and keys from somewhere, and the values from somewhere else.

![[Pasted image 20240130193327.png]]
(English to French) In the transformer decoder block's cross attention, we take the queries and keys from the encoder-learned distillation of our english sentence, and take the values from the input to the decoder. We use our Q and K to form our n\*n attention matrix; every word in our english sentence informs every other word in out english sentence, but the semantic meaning I actually want in my output comes from french. So we take the values we're learning in our french sentence, and multiply by the attention matrix that we're getting from the english sentence.



