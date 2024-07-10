---
aliases:
  - RAG
---
May 22, 2020 (1 month before GPT-3)
[[Meta AI Research]], UCL, NYU -- Authors include [[Douwe Kiela]]
Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
#zotero 
Takeaway: LMs have limitations around hallucinations, provenance, and editing of knowledge. Authors combat this by introducing a retriever+document index into the equation, grounding the generations. Uses [[Dense Passage Retrieval|DPR]] + Wikipedia documents for retrieval, and [[BART]] for generation.

Note that "RAG" is both a specific model/technique as described in this paper, but it's become a general term as well for retrieving text information and placing it into the prompt of a language model, though in those cases, it's often the case that the *retriever*  and *reader* aren't jointly trained.

----

The authors note that there are many shortcomings of the "End-to-End LLM Strategy", including ==hallucinations==, ==inability to update/edit information==, and ==inability to prove provenance of responses==. By incorporating retrieval over a knowledge store, they hope to use the parametric "==generator==" language model for "understanding," and the documents+"==retriever==" (over a vector index of Wikipedia, in this case) for "knowledge," in an effort to ameliorate the three limitations above.

Notes:
- References [[REALM]] and [[ORQA]] as other recently-introduced models that combined (masked) language models with a differentiable retriever.
- The retriever that is used to retrieve relevant Wikipedia documents is [[Dense Passage Retrieval]], from the DPR paper. It's a [[BERT|BERT]]-based [[Bi-Encoder]].
- The generator component is [[BART]]-large, a pretrained seq2seq transformer with 400M parameters. It receives the input *x* concatenated with the retrieved content *z*.
- Starts with a pre-trained "generator" Seq2Seq model ([[BART]]) and a pre-trained "retriever" model (DPR), and jointly fine-tunes them on sequence-to-sequence tasks.
- Proposes two ways in which the generator can use the retriever:
	- ==RAG-Sequence==: In which document(s) are initially retrieved, and the model uses the same document to predict each target token in the output sequence. Model uses the same document to generate the complete sequence; for k retrieved documents, the generator produces an ***output for each document***. Then, the probability of each output sequence is ***marginalized*** (sum the probability of each output sequence in k and weigh it by the probability of each document being retrieved). Finally, the output sequence with the highest probability is selected.
	- ==RAG-Token==: In which each output token can be predicted based on different retrieved documents. Given k retrieved documents, the generator produces a distribution for the next output token for each document before marginalizing (aggregating all the individual token distributions). The process is then repeated for the next token. This means that, for each token generation, it can retrieve a different set of k relevant documents based on the original input *and* previously-generated tokens. Thus, documents can have different retrieval probabilities and contribute differently to the next generated token.
- The authors note that even though grounding models in real knowledge will make them hallucinate less, there's no guarantee that Wikipedia (or any other external knowledge source) will be entirely factual or devoid of bias.
- ==Something that the paper didn't talk about:==
	- If, during fine-tuning, you're also updating the parameters of the *document-encoding* tower of the BERT-based bi-encoder, then you have to re-index all of your documents! I don't recall the paper making it clear, when it talks about how the "retriever" is jointly encoded, whether it's fine-tuning both towers, or if we're just updating the parameters of the query-encoding tower of our Bi-Encoder, which wouldn't require updating the document indexes.

Abstract
> Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, ==their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures==. Additionally, ==providing provenance== for their decisions and ==updating their world knowledge== ==remain open research problems==. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks. We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) -- models which combine pre-trained parametric and non-parametric memory for language generation. ==We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia,== accessed with a pre-trained neural retriever. ==We compare two RAG formulations==, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

Above: 
- I'm a little confused about why they're talking about "nonparametric memory". I guess if you just consider the indexed documents to be the memory, then yeah, it's nonparametric, but the retriever that we use to fetch documents is of course parametric -- it's DPR, a BERT-based Bi-Encoder retriever!
	- Edit: Yeah, the paper later says: "*We refer to the document index as the non-parametric memory.*"


# Paper Figures
![[Pasted image 20240501162150.png]]
Notable that unlike other "modern"/colloquial uses of RAG, in this paper we jointly fine-tune the pretrained LM and the pretrained retriever. Note that their use of MIPS I think just means "Find the documents with the highest dot product with our query."

![[Pasted image 20240501164404.png]]
Above: RAG-T and RAG-S are the RAG-Token and RAG-Sequence methods.

# Non-Paper Figures

![[Pasted image 20240108230651.png]]