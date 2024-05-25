From 2018's *SentencePiece: A simple and language-independent subword tokenizer and detokenizer for Neural Text Processing (Kudo et al)*

Other tokenization algorithms ([[Byte-Pair Encoding|BPE]], [[WordPiece]]) have the same problem, which is that ==they assume that the input text uses *spaces*  to separate words!== -- but not all languages use spaces to separate words.

SentencePiece treats the input as a raw input stream, thus including the space in the set of characters to use. It then uses the BPE or Unigram algorithm to construct the appropriate vocabulary.

Used by models like ALBERT, XLNet, Marian, and [[T5]]
