---
aliases:
  - Microsoft Machine Reading Comprehension
---
November 28, 2016 -- [[Microsoft Research]]
Paper: [MS MARCO: A Human Generated Machine Reading Comprehension Dataset](https://arxiv.org/abs/1611.09268)

A dataset relevant to [[Information Retrieval]] benchmarking and [[Question Answering]] models
~1,000,000 anonymized questions sampled from Bing search queries, and the documents necessary to answer those questions (though some questions might have multiple relevant documents, or none!).

Abstract
> We introduce a large scale ==MAchine Reading COmprehension== dataset, which we name ==MS MARCO==. The dataset comprises of ==1,010,916 anonymized questions==---sampled from Bing's search query logs---==each with a human generated answer and 182,669 completely human rewritten generated answers==. ==In addition==, the dataset contains ==8,841,823 passages==---extracted from 3,563,535 web documents retrieved by Bing---==that provide the information necessary for curating the natural language answers==. A question in the MS MARCO ==dataset may have multiple answers or no answers at all==. Using this dataset, we propose three different tasks with varying levels of difficulty: (i) predict if a question is answerable given a set of context passages, and extract and synthesize the answer as a human would (ii) generate a well-formed answer (if possible) based on the context passages that can be understood with the question and passage context, and finally (iii) rank a set of retrieved passages given a question. The size of the dataset and the fact that the questions are derived from real user search queries distinguishes MS MARCO from other well-known publicly available datasets for machine reading comprehension and question-answering. We believe that the scale and the real-world nature of this dataset makes it ==attractive for benchmarking machine reading comprehension and question-answering models==.
