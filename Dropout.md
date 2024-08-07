A [[Regularization]] technique used in ML (mostly NNs) to prevent overfitting by randomly "dropping out" (setting to zero) a fraction of the neurons during the training process.

During each training iteration, a certain percentage (==dropout rate==) of neurons in the network are randomly selected and temporarily removed, along with their incoming/output connections. By dropping different sets of weights in each iteration ((epoch?)), the network learns to distribute its weights more evenly across all of the neurons, making it more robust and less likely to overfit. ==During the inference phrase, dropout is not applied== -- Instead, the weights of the neurons are ==scaled== down by the dropout rate to account for the fact that they were dropped out during training.


![[Pasted image 20240701101449.png]]
