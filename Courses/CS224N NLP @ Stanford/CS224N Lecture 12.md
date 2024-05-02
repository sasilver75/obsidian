#lecture 
Link:https://www.youtube.com/watch?v=NcqfHa0_YmU&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=12

---
# Lecture 11: Question Answering
Lecturer: Danqi Chen (Past the halfway point: Week 6!)
(One of the foremost researchers in Question Answering and well known for being one of the authors of the [[RoBERTa]] paper, SpanBERT, and more! She was once the head TA of this course)

Lecture Plan:
1. What is QA? (10 mins)
2. Reading comprehension (50 mins)
	- How do we answer questions over a SINGLE PASSAGE of text?
3. Open-domain (textual) question answering (20 mins)
	- How do we answer questions of a LARGE COLLECTION of documents?

![[Pasted image 20240411204536.png]]

![[Pasted image 20240411204634.png]]
![[Pasted image 20240411204743.png]]

Lots of practical applications of Question-Answering!
- Smart Speaker/SIRI/Alexa type shit
- Search engines
- IBM Watson
- ...

![[Pasted image 20240411205015.png]]
Complex systems doing many NLP tasks!

![[Pasted image 20240411205056.png]]
Very different today from the Watson era


![[Pasted image 20240411205159.png]]
==Knowledge base question-answering==; answer questions over a large database of unstructured information!

Given a question
- Convert into some logical form that can be executed against a database to give us a final answer.

![[Pasted image 20240411205235.png]]
==Visual question-answering==
- An active CV/NLP field

![[Pasted image 20240411205419.png]]


![[Pasted image 20240411205846.png]]
Rephrasing difficult problems as a reading comprehension problem (perhaps breaking them down into some sort of CoT type thing)

![[Pasted image 20240411210156.png]]
[[SQuAD]]: Stanford Question Answering Dataset (2016)
- 100k annotated (passage, question, answer) triplets!
	- Important: Answers are a short of text in the passage. ==This is a limitation! NOT all questions can be answered this way!==

![[Pasted image 20240411210421.png]]

So if we  can do other tasks like NER, Relation Extraction by sticking something on top of BERT and finetuning it, or doing it as Q&A -- does one method work better than the other, and by how much?


![[Pasted image 20240411211016.png]]

# BREAK
Honestly I'm going to skip this lecture. It's not the most interesting subject to me, and the teacher, while very intelligent/proficient, isn't a native english speacher, and it's a little bit of a chore.








