![[Pasted image 20240410180558.png]]

We freeze all the parameters of the pretrained network itself, and never change any of them. Instead, we make a bunch of fake pseudo-word vectors that we prepend to the beginning of a sequence, and we just train them!
- These would have been inputs to the network, but we specify them as parameters, and just train the values/parameters of the fake words.
- This keeps all the generality of the model params, and is easier to do than finetune the entire network.