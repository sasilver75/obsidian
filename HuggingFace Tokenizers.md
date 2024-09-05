Offers access to tokenizers used by popular transformer-based models, like [[BERT|BERT]], [[GPT]], and [[T5]], which ensures crucial consistency between tokenization used during pre-training and fine-tuning.

re: [[Natural Language Toolkit|NLTK]], [[spaCy]], [[HuggingFace Tokenizers]]:
> "Each of these providers offers unique strengths and features, catering to different needs and use cases within the NLP domain. NLTK's simplicity makes it a great choice for educational and exploratory purposes, while spaCy's performance and linguistic awareness make it suitable for production-ready applications. Hugging Face Tokenizers, on the other hand, excel at handling the complexities of advanced NLP models, ensuring seamless integration with state-of-the-art architectures."


```python
from transformers import BertTokenizer

# Create a BerTokenizer instance using an identifier on the HF hub
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# use the Tokenizer.tokenize(...) method to tokenize text
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.tokenize(text)
```
This specific HuggingFace tokenizer uses a mode advanced tokenization approach called [[WordPiece]] tokenization, which is a type of subword tokenization. The resulting vocabulary contains a mix of full words and common subword units. When presented with new text

---
Aside on [[WordPiece]] tokenization:
1. Start with a vocabulary of individual characters.
2. Iteratively find the most common pair of tokens (either characters or subwords) and merges them into a new subword token.
3. Process continues until the desired vocabulary size is reached

So this is very similar to [[Byte-Pair Encoding|BPE]], in that they start from characters and perform merges. See more [here](https://www.reddit.com/r/MachineLearning/comments/rprmq3/d_sentencepiece_wordpiece_bpe_which_tokenizer_is/)

---


