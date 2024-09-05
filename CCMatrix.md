November 10, 2019
[[Meta AI Research]] (*Schwenk et al.*)
[CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB](https://arxiv.org/abs/1911.04944)



See previous: [[CCAligned]]
![[Pasted image 20240722174105.png]]


Abstract
> We show that margin-based bitext mining in a multilingual sentence space can be applied to monolingual corpora of billions of sentences. We are using ten snapshots of a curated common crawl corpus (Wenzek et al., 2019) totalling 32.7 billion unique sentences. Using one unified approach for 38 languages, we were able to mine 4.5 billions parallel sentences, out of which 661 million are aligned with English. 20 language pairs have more then 30 million parallel sentences, 112 more then 10 million, and most more than one million, including direct alignments between many European or Asian languages.  
> To evaluate the quality of the mined bitexts, we train NMT systems for most of the language pairs and evaluate them on TED, WMT and WAT test sets. Using our mined bitexts only and no human translated parallel data, we achieve a new state-of-the-art for a single system on the WMT'19 test set for translation between English and German, Russian and Chinese, as well as German/French. In particular, our English/German system outperforms the best single one by close to 4 BLEU points and is almost on pair with best WMT'19 evaluation system which uses system combination and back-translation. We also achieve excellent results for distant languages pairs like Russian/Japanese, outperforming the best submission at the 2019 workshop on Asian Translation (WAT).