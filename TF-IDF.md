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

![[Pasted image 20240426134723.png|200]]
==With TF-IDF, we're looking for words that are *truly* distinguishing indicators of a document.== 
- TF-IDF reaches *maximum values*  for terms that are *very frequent* in a small number of documents
- TF-IDF reaches *minimum* values for terms that are *very infrequent* in a *very large* number of documents (but it's also very small for terms that are very frequent in almost every document, like "and")


To calculate relevance scores for a given query with multiple terms, we do a 
![[Pasted image 20240426134914.png]]
Where here, Weight would be the TF-IDF score.

