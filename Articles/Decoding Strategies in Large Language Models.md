https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html
A Guide to Text Generation from Beam Search too Nucleus Sampling

---

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load our Tokenizer and Model (in eval mode, on GPU)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval()

# Let's tokenize the following text
text = "I have a dream"
input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
# tensor([[  40,  423,  257, 4320]])

# I think this is asking to generate 5 additional tokens on the end, in sequence.
outputs = model.generate(
	input_ids,
	max_length=len(input_ids.squeeze())+5
) # tensor([[  40,  423,  257, 4320,  286,  852,  257, 6253,   13]])
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
# 'I have a dream of being a doctor.'
```

There's a common misconception that LLMs like GPT-2 directly produce text -- but this isn't the case! Instead, LLMs themselves calculate logits, which are just scores assigned to every possible token in the vocabulary. These logits are converted to a probability distribution over the vocabulary using a softmax function that can be considered as separate from the mode.

![[Pasted image 20240627221404.png]]

Autoregressive models predict the next token in a sequence based on the preceding tokens. P(of | I have a dream) = 17. 
![[Pasted image 20240627223103.png|400]]
We do this to calculate the conditional probability for every token in the vocabulary.

So how do sample from these probabilities to generate text? There are many options.


----


# Greedy Search
- [[Greedy Decoding]] is a decoding method that takes the most probable token at each step.
- This might sound intuitive, but it's important to note that greedy search is short-sided: It only considers the most likely token at each step without considering the overall effect on the sequence.
- It's fast and efficient, since it doesn't need to keep track of multiple sequences like [[Beam Search]] does.


