January 31, 2024
[[Allen Institute]]
[Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159)
...
Significance: ...

-----


An English-only 3T token corpus representing a diverse mixture of web content, scientific papers, code, public-domain books, social media, and encyclopedic materials.

They also open-source their data curation toolkit to enable further curation.

==This would be an incredibly useful paper for me to dive in on==

Released together with [[OLMo]] (separate papers)

Abstract
> ==Language models== have become a critical technology to tackling a wide range of natural language processing tasks, yet many details about how the best-performing language models were developed are not reported. In particular, ==information about their pretraining corpora is seldom discussed==: commercial language models rarely provide any information about their data; even open models rarely release datasets they are trained on, ==or an exact recipe to reproduce them==. As a result, it is challenging to conduct certain threads of language modeling research, such as understanding how training data impacts model capabilities and shapes their limitations. To facilitate open research on language model pretraining, we release ==Dolma==, a ==three trillion tokens English corpus==, built from a ==diverse mixture of web content, scientific papers, code, public-domain books, social media, and encyclopedic materials==. In addition, ==we open source our data curation toolkit== to enable further experimentation and reproduction of our work. In this report, ==we document Dolma, including its design principles, details about its construction, and a summary of its contents==. We interleave this report with analyses and experimental results from training language models on intermediate states of Dolma to share ==what we have learned about important data curation practices, including the role of content or quality filters, deduplication, and multi-source mixing==. Dolma has been used to train OLMo, a state-of-the-art, open language model and framework designed to build and study the science of language modeling.

# Paper figures

# Other figures
![[Pasted image 20240429114224.png]]
![[Pasted image 20240429114329.png]]
![[Pasted image 20240429114346.png]]
(This talk was in April 2024 @ Cornell Tech, but they don't include the 30T RedPajama-v2)
![[Pasted image 20240429114511.png]]
Showing the most common n-grams in C4