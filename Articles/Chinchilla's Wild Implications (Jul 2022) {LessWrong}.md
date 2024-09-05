---
tags:
  - article
---

Link: https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications

-----

This post is about the language model scaling laws as defined in the [[DeepMind]] paper that introduced [[Chinchilla]].

The implications:
1. ==Data, not size, is the currently active constraint on language modeling performance==. Current returns to additional data are immense, and current returns to additional model size are miniscule.
	- In fact, most recent landmark models were wastefully big, given compute.
	- If we have enough data, there's not need to train 1T parameter models.
2. ==The literature is unclear on how much high-quality text data is *actually* available for training==. We may be "running out," or we may not be -- the literature is too vague.
3. The *entire* ==available quantity of data in highly-specialized domains like code is woefully tiny==. It would be fabulous if more such data were available.


# 1. The Scaling Law

- The paper fits a scaling law for Language Model loss `L` as being a function of the model size `N` and data size `D`. Again, ==the paper fits loss as a function of model size and data size.==

![[Pasted image 20240207145739.png]]
- The first term depends only on model size, the second only on data size, and the third is some constant.
- Consider:
	- An "infinitely big" model trained on "infinite data" would be able to achieve loss `E`. In order to get the loss for a *real* model, we have to add two "corrections":
		- That the model was only trained with `N` parameters, not an infinite number.
		- That the model was only trained on `D`-sized data, not an infinite amount of data.

![[Pasted image 20240207145913.png]]

With this parametrization, DeepMind fit it on the MassiveText dataset:
![[Pasted image 20240207145940.png]]


If ==we take this fit model and apply it to the the [[Gopher]] model== with 280B params, trained on 300B tokens of data:

![[Pasted image 20240207154105.png]]
Above:
- What jumps out is that the "finite model" term is *tiny*; This means that Gopher's parameter count *might as well* be infinity! There's very little left to gain on the front of increasing the number of parameters in that model.
- Meanwhile, the "finite data" term is *not* tiny! Gopher's training data size is very much NOT infinity, as we can go a long way by making it bigger!


[[Chinchilla]] is a model with the same training compute cost as [[Gopher]], but with the parameters/data allocated more evenly across the two terms in the equation. It's 70B params, trained on 1.4T tokens of data. ==This is the Gopher-fit model applied to the Chinchilla model:== 
![[Pasted image 20240207154247.png]]
==We can see that Chinchilla achieved roughly the same loss as Gopher by using only 70B params rather than 280B params, simply by increasing the amount of data from 300B tokens to 1.4T tokens==
- Recall that ==it was conventional at the time to train ALL large LMs on roughly 300B tokens of data -- because GPT-3 did it, any everyone else followed!==
	- If we trust our equation, this *entire line of research* could have _never_ beaten Chinchilla, no matter _how big_ the models got!

People put immense effort into training huge models, and were even working on bigger ones, but none of this, in principle, could ever get as far as Chinchilla did!

![[Pasted image 20240207154555.png]]
Above:
- Only [[PaLM]] is remotely close to Chinchilla! PaLM is a HUGE model -- it's the largest considered here. Everyone writing about PaLM mentions that it has 540B parameters, and the PaLM paper does a LOT of experiments on the differences between the 540B PaLM and smaller variants of it.
	- PaLM isn't competitive with Chinchilla because it's *big* -- the MT-NLG model is almost the same size, but it's trapped in the pinkish/purplish zone with Gopher and the rest. PaLM is competitive with Chinchilla only because it was trained on more tokens (780B) than the other non-Chinchilla models.


![[Pasted image 20240207154945.png]]
- You can see again that there aren't many more gains to eke out from the model size. PaLM's gains came with a great cost, though -- it used WAEY more training compute than any previous model, and its size means it also takes a lot of inference compute to run!
![[Pasted image 20240207155039.png]]
Above: See that Chinchilla and PaLM have similar losses, but PaLM takes much more compute to get to the same point. We spent *all that compute* on PaLM, and only got a slight edge over Chinchilla! Given PaLM's compute, we should have used *more data* and trained a *smaller model*.

![[Pasted image 20240207155333.png]]

The optimal amount of data for PaLM would have been around 6.7T tokens, around ~.4.8x as many tokens as used by Chinchilla! And the resulting model would only have 63B params, rather than 540B params!

==Wait... do we have 6.7T high-quality text tokens lying around?==


# 2. Are we running out of data?
- It's very difficult to find an answer to this question!
- LM papers are meticulous about `N`, model size. There's been a lot of smart discussion about the hardware and software demands of training high-N models. 
- But meanwhile:
	- Everyone just dumbly trained their models on 300B tokens for no particular reason, just because GPT-3 did. This paper showed how hilariously wasteful that is.
	- Papers *rarely* do scaling analysis that vary *data size*! When people hear "LM Scaling," their brains generally just map that to "increasing parameter size."
	- Papers basically never talk about what it would take to scale their *datasets* up by 10x or 50x!
	- The data collection sequences of LM papers tend to be vague and slapdash, often failing to answer basic questions.

### Web Scrapes
- If you just was *a lot of text*, the easiest way to get it is from web scrapes like [[Common Crawl]] -- but ==these are infamously full of garbage -- if you want to train a good LM, you probably want to aggressively filter them for quality!==
### MassiveWeb
- The training dataset used for Gopher and Chinchilla is called MassiveText, and the web-scrape portion of it is called MassiveWeb. 
	- This data originates from a mysterious, unspecified web-scrape, which is funneled through a series of filters, including quality heuristics and an attempt to only keep English text.
	- MassiveWeb is 506B tokens; could it be made bigger, by scaling up the original web scrape? We really don't know anything about it.
### The GLaM/PaLM web corpus
- PaLM used a different web-scrape corpus. It doesn't say anything about the original scraping process, only describing the quality filtering they did (and not in much detail!).
> GLaM paper:"There's a large amount of very high-quality textual data on the web, but there isn't an infinite amount. For the corpus-mixing proposition chosen for PaLM, data begins to *repeat* in our subcorpora after 780B tokens, which is why we chose that as the endpoint of training! It's unclear how the "value" of repeated data compares to unseen data for large-scale language model training!"
- Yikes, it seems like as you scale your dataset size, you get a lot of repeats, and they aren't sure about the effects of that.
- ==It seems like even the vast web data resources available to Google Research are starting to strain against the data demands of large LMs! Is that plausible?== 

### Domain-specific corpora

#### Code
- If you want code, it's on GitHub. We've already more-or-less exhausted GitHub! It's been scraped a few times with different kinds of filtering, yielding broadly similar data sizes:
	- [[The Pile]]'s scrape has 631GBGB of text, and 299B tokens
	- The [[MassiveText]] scrape had 3.1TB of text, and 506B tokens
	- The [[PaLM]] scrape had only 196GB of text (unsure how many tokens)
	- The [[Codex]] paper's scrape was python-only and had 159GB or text

All of these scrapes contained a large fraction of the total code available on GitHub.... generously, ==there might be ~1T tokens of code out there, but not vastly more than that.==

#### Arxiv
- If you want to train a model on advanced academic research in physics or math, go to Arxiv! 
- Arxiv was about half the training data for the math-problem-solving LM [[Minerva]].
- ==We've already exhausted Arxiv -- both the Minerva paper and the [[The Pile]] use basically all of Arxiv, with a measly 21B tokens.==


#### Books
- In [[The Pile]], "books" means the Books3 corpus, which means "all of Bibliotik"; it contains 196,640 full-text books, amounting to only 27B tokens.
- In [[MassiveText]], a mysterious subset called "books" has 560B tokens -- that's a LOT more than The Pile! Is this... all the books in the world? Who knows?
- In the GLaM/PaLM dataset, an equally-mysterious subset called "books" has 390B tokens!
- ==It's hard to know what to make of these datasets; are they "basically all the books in the world," or just some subset that an engineer pulled at one point in time?==


#### All the data we have
- ==He tried to add up the tokens that he knows about, and the the of ~3.2T tokens, or about 1.6x the size of MassiveText, and about 35% of the data we would need to train an optimal PaLM model.==


What is compute? On a further barrier to scaling
- Here's another important comment from the PaLM paper's Chinchilla discussion:
> If the smaller model were trained using fewer TPU chips than the larger model, this would proportionally increase the wall-clock time of training, since the total training FLOP count is the same. If it were trained using the same number of TPU chips, it would be very difficult to maintain TPU compute efficiency without a drastic increase in batch size.

In LM scaling research, all "compute" is treated as fungible -- there's one research, and you can spend it on params and steps, where ==compute = params * steps==

To scale up data, `D`, you have to either:
- Take more optimization steps (an inherently serial process, which takes linearly more time as you add data).
- Increase the batch size (which tends to degrade model quality behind a certain critical size, and current high-N models are already pushing against that limit.


-----

#### Interesting Comments

> ==One important implication is that pure AI companies like OpenAI, Anthropic, Conjecture, Cohere are likely to fall behind companies with access to large amounts of non-public-internet data like Facebook, Google, Apple==. Email and messaging are especially massive sources of "dark" data, provided they can be used legally and safely. Taking just email, something like 500 billion emails are sent daily, which is more text than any LLM has ever been trained on.
> ...
> Another implication is that ==federated learning, data democratization efforts, and privacy regulations like GDPR are MORE likely to be critical levers on the future of AI than previously thought.==

In response:
> Another implication is that ==centralized governments with the ability to aggressively collect and monitor citizen's data, like China, could be major players==.


























