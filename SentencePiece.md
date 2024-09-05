From 2018's *SentencePiece: A simple and language-independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al)*

Considers text as a sequence of unicode characters, and replaces spaces with a special character \_. Used in conjunction with the Unigram algorithm, it doesn't even require a pre-tokenization step, which is useful for languages were the space character is not used. The other main features is *reversible tokenization* -- since there's no special treatment of spaces, decoding the tokens is simply done by concatenating them and replacing any \_s with spaces, resulting in the normalized text. Other tokenizations involve removing (eg) repeated spaces, so tokenization isn't reversible.


Other tokenization algorithms ([[Byte-Pair Encoding|BPE]], [[WordPiece]]) have the same problem, which is that ==they assume that the input text uses *spaces*  to separate words!== -- but not all languages use spaces to separate words.

SentencePiece treats the input as a raw input stream, thus including the space in the set of characters to use. It then uses the BPE or Unigram algorithm to construct the appropriate vocabulary.

Used by models like ALBERT, XLNet, Marian, and [[T5]]
