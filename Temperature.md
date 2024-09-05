A hyperparameter that can be added to the softmax function to change the distribution of results.

> "Temperature is the blood-alcohol content of your language model"
> - John Berryman, MLE @ Github

![[Pasted image 20240411155632.png]]

 This temperature doesn't affect the monotonicity of the distribution; if word A has higher probability than word B previously, then after the adjustment A will still have a higher than B, but ==their relative difference will change!==
	- Temperature $\tau$ > 1  --> Our distribution becomes more uniform, producing more diverse output!
	- Temperature $\tau$ < 1 --> Our distribution becomes more spiky, producing more "boring" output, since probability is concentrated on top words. In the extreme case, it turns into effectively greedy decoding, producing basically one-hot vectors!