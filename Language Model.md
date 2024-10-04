A predictive model aimed at predicting the next token in a sequence of tokens.


---
Fun fact: Language models were initially interesting for the problem of [[Automatic Speech Recognition]], where we wanted to disambiguate two possible parsed sentences, given some acoustic data.

"I understand you ==like your== mother"
"I understand you ==lie cure== mother"

The above examples acoustically sound very similar, but with an understanding of the distribution of written tokens, we have a better idea that the top example is more likely.

---