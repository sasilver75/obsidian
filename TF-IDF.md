---
aliases:
  - Term Frequency Inverse Document Frequency
---
![[Pasted image 20240425210755.png]]
==TF-IDF== is a way of massaging those values to get more information about relevance
- Begin from a corpus of documents $D$ 
- Term frequency of word, given document, is the number of times the word appears in the document, divided by the length of the document
- Document frequency of words is just counting the number of document that contain the target word, regardless of how many times it occurs
- Inverse document frequency is the log of the total size of our document corpus divided by the document frequency value above.
- [[TF-IDF]] then is just the product of the TF and IDF values.