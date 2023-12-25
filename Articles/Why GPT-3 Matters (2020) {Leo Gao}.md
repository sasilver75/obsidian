https://bmk.sh/2020/05/29/GPT-3-A-Brief-Summary/
By Leo Gao (https://twitter.com/nabla_theta)

![Number of Parameters of GPT-3 compared to previous models. (<a href='https://www.willstats.com/'>Edited by WillStats</a>, <a href='https://arxiv.org/abs/1910.01108'>Original 1</a>, <a href='https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/'>Original 2</a>)](https://bmk.sh/images/gpt3/title.png)
*Number of Parameters of GPT-3 compared to previous models*

The sheer scale of GPT-3 was hard to overstate -- it's an entire *order of magnitude* larger than Microsoft's 17B Turing-NLG model.
- Loading the entire model's weights in fp16 would take something like 300GB of VRAm, not even including the gradients.

With GPT-3's massive size came massive generalization ability! GPT-3 is competitive on many benchmarks *without even tuning on the target task!*
- The full 72-page paper contains evaluations on many NLP datasets.

Perhaps the most impressive part is that at even such a large scale, the model still scales smoothly in performance, instead of plateauing -- implying that still longer models would perform *even better.*

While GPT-3's ~170B model isn't that much deeper than Microsoft's 17B Turing-NLG model,  but it's nearly 3x wider!
- Since parameter count scales approximately proportional to the square of the hidden layer size (since the layers are densely connected), that explains where the large parameter count in GPT-3 comes from.

GPT-3 also has double the context size, at 2048 tokens.
- Some models at the time had even longer context, like Transformer-XL and Reformer.

GPT-3 uses sparse attention layers in every other layer... (I believe in an attempt to reduce the quadratic cost of attention, as we compute the attention weighting of every token to every other token).

GPT-3 reuses the BPE tokenization of GPT-2.

Overall, GPT-3 is essentially just a massive version of GPT-2 (and we mentioned that even then, it seems like a larger model would perform even better!)

### Training Data
![Weighted Training Data (<a href='https://arxiv.org/abs/2005.14165'>Source</a>)](https://bmk.sh/images/gpt3/tdata.png)
- The training data is a reweighted mix of [[CommonCrawl]], WebText2, Book1, Book2, and Wikipedia!
- Some of these components (like Wikipedia) were seen more than 3 times during training -- others, like the massive Common Crawl, had less than half of its data seen.
- English constituted ~93% of the dataset.


### Evaluation
![Zero-, One-, and Few-shot performance of GPT-3 scaling with parameter count (<a href='https://arxiv.org/abs/2005.14165'>Source</a>)](https://bmk.sh/images/gpt3/perf-small.png)
- Interestingly, all of the Zero, One, and Few-shot abilities all scaled with parameter count!
	- The ability to *infer the task* from just one or a few examples is a massive step forward in generalization! Previous models all relied on specific-task tuning, but GPT-3 can be "tuned" by giving it instructions/examples *in plain english,* in the prompt! 
		- The paper didn't even *attempt* to fine-tune the actual model on specific tasks.

One crucial conclusion is that in all tests, performance continues to get better with larger models ... whereas fine-tuning models of a certain side often improved performance on a narrow set of tasks, but risked [[Catastrophic Forgetting]] and [[Overfitting]].

Abilities:
- GPT-3 can also finely do some arithmetic.
- People are unable to distinguish GPT-3-generated news stories from real ones, only exacerbating the ethical concerns already raised by GPT-2. 
- Authors note that GPT-3 is male leaning. Similar issues appeared for race and religion.


### Downstream Applications
- GPT-3 has already been used for a bunch of applications
	- writing code
	- turning natural language commands into shell commands
	- chatting with famous scientists
	- answering medical questions
	- writing copypastas
	- summarizing passages for second graders
	- writing poetry
- These were all done with the *same exact model* trained *only* on the modeling text! It's a pretrained LLM that has just been "asked nicely" to do different things.

### Conclusion
- Why does GPT-3 matter, if it didn't beat SOTA across all benchmarks and was super expensive to train? 
	- It's impressive because it's doing reasonably well on tasks *that it's never even seen!* And sometimes tasks that *aren't even anticipated by the developers of hte model!*
	- Additionally, instead of reaching a point of diminishing returns, GPT-3 shows that the trend of larger models performing better continues for at least another order of moagnitude -- with no signs of stopping!

























