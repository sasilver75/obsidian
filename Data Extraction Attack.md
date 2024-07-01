
## Data Extraction Attacks
- Neural networks often unintentionally memorized aspects of data!
- Given a model, can we figure out what points were used in the training dataset, without having a specific datapoint in mind? Given $M$ , find some $X \in D$ 

We can attack it with prompt engineering:
`"My social security number is: "` ... will the model output something sensitive?

Carlini et al, 2021 (Can't remember attack name):
1. Sample a bunch of data from the model
2. Do a membership inference attack (figure out which of the generations were in the training set)
![[Pasted image 20240701135236.png]]
In this attack, we use a pretty straightforward metrics-based approach. We use $p(x_i|x_1,...,x_i-1)$ , specifically [[Perplexity]] as a score to evaluate the generations (the authors experiment with multiple metrics).
$exp((-1/n)\sum_{i=1}^n{log(p(x_i|previous))})$ 

The authors used this to attack (in 2021) GPT-2, and found that they could extract hundreds of memorized training examples from GPT-2, and confirmed this with the authors at OpenAI.

