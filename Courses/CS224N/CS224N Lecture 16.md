#lecture 
Link:https://www.youtube.com/watch?v=5vfIT5LOkR0

Subject: Multimodal Deep Learning

------
![[Pasted image 20240417000000.png]]

Today, Chris introduces us to [[Douwe Kiela]], the guest lecture for this lecture ([[Meta AI Research|FAIR]], where he invented [[Retrieval-Augmented Generation]], [[HuggingFace]]), now a Stanford Professor

------

Trying to keep things to what's useful

[[Multimodal|Multimodality]]: "Having, involving several modes, modalities, or maxima." In our case, focusing on NLP, it's *text plus one other modality (images, speech, audio, olfaction, others).* We'll mostly focus on images as the other modality.


## Why does MultiModality matter?
1. Faithfulness: Human experience itself is very multimodal
2. Practical: The internet and many applications is multimodal
3. Data efficiency and availability
	- We're sort of running out of data on the text side; to continue scaling, we need more tokens from other modalities!
	- Multimodal data is rich and "high bandwidth"
	- [[Yann LeCun]]: "\[Language is\] An imperfect, incomplete, low-bandwidth serialization protocol for the data structures we call thoughts."

![[Pasted image 20240417002242.png]]
Above: There are many ways to do intermodal fusion!

![[Pasted image 20240417002349.png]]
1. Combine modalities at beginning ("Early fusion")
2. Treat them separately and then later combine them ("Middle fusion")
3. Combine just the scores/logits ("Late fusion")
	- This is what we would now call contrastive models, eg [[CLIP]]

![[Pasted image 20240417002535.png]]
Just want to make sure that we rank the blue things higher than all otehr alternatives; nothing special about this architecture that was invented here, but it was cool because it was transformers, and transformers all the way (text: transformer, images: ViT) -- Trained on lots and lots of data.
- Alec Radford is a genius at creating large and high quality datasets.

==The speaker (Douwe Kiela) thinks that the CILP paper is one of the best papers ever written in our field -- extremely thorough and worth a read!==


![[Pasted image 20240417002825.png]]
Quickly after CLIP, a paper from Google called ALIGN, which really did the same thing and threw more images/text at it. 1.8B pairs instead of 300M, got a better model. Google.

An organization called [[LAION]] has started an open-source collective to create high-quality image-text pairs! (eg LAION 5B)
- Used to train [[Stable Diffusion]]

(Pick back up at 28:00)

Early/Mid fusion (Rather than late-fusion) is really what you should be understanding if you're going to go into research.

## Multimodal Foundation Model section

![[Pasted image 20240417134753.png]]
[[Bidirectional Encoder Representations from Transformers|BERT]]: How do we make this multimodal?
- Student suggestion: You could take the ConvNet features and classifier feature from BERT, concatenate them, and then classify a Cat (or whatever)

![[Pasted image 20240417134959.png]]
Just take your features and plug them into your TRansformer model, and try to recover the features -- this is probably the simplest way to do it.

This is what we call a "==Single Stream architecture==" -- Concatenating the original features and putting them through the Transformer

![[Pasted image 20240417135116.png]]
Contrast that with ==two different streams (Dual Stream)==, as did ViLBERT, where you have two parallel transformers, but at every layer you give them [[Cross-Attention]]

![[Pasted image 20240417135152.png]]
There were obviously many of these types of models that came out in ~2019 or so

You can also do something even dumber (this is the speaker's paper):
![[Pasted image 20240417135234.png]]
- Take the image, put it through a ResNET, then do some pooling on the feature maps, and finally give them to BERT

Visual BERTs: PixelBert
![[Pasted image 20240417135338.png]]
Going to the pixel level completely (Basically the same as MMBT above, but do a multimodal pretraining step)


UNITER
![[Pasted image 20240417135428.png]]


ViLT
![[Pasted image 20240417135433.png]]
The first instance where we're completely gone from ConvNet features; we don't do any preprocessing on the images (no region processing, no backbone -- we just have patches of the image, we flatten those patches, and we pump them through the Transformer straightaway -- this is sort of BERT and ViT together in one model, and worked very well.)

![[Pasted image 20240417135522.png]]
Distinctions:
- What is the TEXT encoder you use?
	- BERT, RoBERTA, ?
- What is the REGION encoder you use?
	- Region CNN (RCNN), ResNet, ViT
- What kind of fusion
	- Single stream, Dual stream
- Pretraining tasks
- Datasetes

![[Pasted image 20240417135652.png]]
It turns out if you take all of these little model inventions and train all these different models on exactly the same data in the same way, it turns out that they're all basically the same -- it's wasted effort to create fancy new small architecture modifications -- ==it's alll about the data!==


![[Pasted image 20240417135756.png]]
Paper from speaker, trying to take multimodality to the limit.
- Instead of language/image, where we want to go is having one model to rule them al -- one that can consume data from all sorts of modalities and synthesize across all these modaliities and learn useful information.
PMD Dataset:
- 70M image-text pairs from public sources
- If you create all datasets that have image-text that's publicly available...

![[Pasted image 20240417135938.png]]

((Reminds me sort of as multimodal T5))

How does FLAVA work?
- Image encoder, encode image as patch; Do masked image modeling (same as masked langauge modeling)
- On the other side, do masked lanugage modeling on the language, and have a multimodal part where all of this information gets combined
- Have a global contrastive loss, like a CLIP -- all transformers, all the way down.

![[Pasted image 20240417140054.png]]
A nice example of where we're probably going to go in the near future


We see also:
- Everyone cares about generative models -- there's a trend where we want to move from contrastive/discriminative to the richer thing of generating images/sequences.

[[SimVLM]]: Generate/complete captions
![[Pasted image 20240417140211.png]]

[[CoCa]] Cotrastive Captioner (Yu et al, 2022) -- Current SoTA as of talk (Apr17 2024 - 6 months)
Best of both contrastive and generative worlds:

![[Pasted image 20240417140248.png]]

Frozen (Tsimpoukelli, Menick, Cabi et al, 2021)
- One of the interesting things you can do with LMs is keep them frozen and learn how to project into the language model space
- Kind of like MMBT but with a better LLM (T5) and a better vision encoder (NF-ResNet)
- ==You get multimodal few shot learners!==
	- ((This is very cool! This is an X, and this is a Y, now what's this picture of?))
![[Pasted image 20240417140339.png]]

[[Flamingo]] (Alaryac et al, 2022) Deepmind
- 80b param model based on Chinchilla
![[Pasted image 20240417140435.png]]
Gets you a much more powerful model, because you can be generative over lots of different images.

Started off with simple transformers, and now we're at something pretty complicated:
![[Pasted image 20240417140635.png]]
Perceiver Resampler: We have a bunch of different images that we featurize, and now we need to cmopress the information (because we sometimes have 3 images, sometimes 5 images -- we need to compress so that it's unifrom and ready for consumption by the next part of the model)

Gated Cross-Attention
![[Pasted image 20240417140714.png]]
You do this before your frozen Language Model layer; you have a frozen Chinchilla language model and learn to modulate the information gonig into that LM; you propagate it with gradients all the way back, you just don't update the LM -- How do I design my signal so that my LM can do the most with it! (We do it before the layer, rather than after)

[[BLIP]], [[BLIP 2]]
![[Pasted image 20240417141043.png]]
In Falmingo, you have a lot of moving parts, but you can take this to the full extreme you try freeze almost everything, and you just try to learn a mapping between your image encoder and your language model (or your image encoder and encoder/decoder architecture) -- and you learn this projection between the two.
They experiment with OPT as the LM and Flan-T5 as the other one
Gives great captions without direct supervision on the captions themselves; show you the power of langauge models in general!



![[Pasted image 20240417141224.png]]
"Generate a rationale for something that might be the case!"

![[Pasted image 20240417141316.png]]
KOSMOS-1 did really great work on Raven's Progressive Matrices, which are very hard visual problems! This seems to show that we're making progress!

# Beyond images: Other modalities
- Speech/audio
	- Could easily do another full lecture just on this topic!
	- Recent cool example; [[Whisper]], trained on 680,000 hours of multilingual multitask data (again, [[Alec Radford]] being good at curating large datasets)
	- We can also just treat audio as vision 😜 where we use a CNN on an audio spectrogram
- Video and text and audio
	- Images naturally extend to video (which also includes audio!)
	- Subsample the frames to get the most useful information from the video, since adjacent frames in a 120fps video probably don't have a lot of additional information.
	- See: MERLOT and MERLOT Reserve
- Scent/Olfactory!
	- Fuck it, it works!


-------
20VC interview with [[Douwe Kiela]]

Problems with ML:
- We don't know why they'er saying what they're saying
- Theres' compliance issuse -- how to remove information from them (eg GDPR)
- We can't revise information
- We can't keep it up to date
- Data privacy issues (eg HIPPA, private company information)

How did you make your way into ML/NLP?
- High school in the Netherlands, fascinated by computers
- By the time he went to go to college, he thought he knew everything about CS, so he decided to study philosophy instead; a radical departure from what he'd bee interested in at the time.
- At some point, it became clear that he had to start making money, and Phil isn't a real job... so he did some Logic...and then some Computer Science @ Cambridge after all.
- That's where he started doing NLP; one of his internships was at MSR in NY with a great mentor (One of Lecun's henchmen, invented SGD)
- When they started FAIR, he joined it out of his PhD, which kicked off Douwe's career.

5 years at [[Meta AI Research|FAIR]]
- Biggest takeaways: Had a bunch of lovely people that he learned so much from - mostly how to focus your research direction. Having a clear, real-world application for the research you're doing makes it much more valuable than going off on a tangent and being too far ahead of the rest of the field.
Then spent time at [[HuggingFace]]
- "A fascinating company"; Was at Meta, looking for something new, thought about a startup but figured he needed more experience at a successful AI startup -- HF was that.
- What impressed me is how good they are at marketing, branding, community building -- everyone loves the company, and he still doesn't understand how they do that.

Okay, then where does [[Contextual]] come from? (Which he cofounded with a friend from FB, HF)
- Belief that models as they exist aren't ready for primetime
	- ==Hallucination==
	- ==Lack of Attribution==
	- ==Compliance isusues==
	- ==Can't revise information==
	- ==Can't keep information up to date==
	- ==Data privacy issues==
	- ==Models are still quite inefficient; can be faster==
- At Contextual, building a "DIFFERENT KIDN OF LM", from first principles for enterprise use cases
	- Trying to be a bit smarter
	- Arch is based on [[Retrieval-Augmented Generation]] , where you can decouple compute and memory (which can be updated/revised on the fly).

"Hallucinations are a Feature, not a Bug" - Emad
- Douwe thinks it's a great quote, but it's a bit more naunced than "Y/N" -- if you want the model to be very creative, you want it to hallucinate! A spectrum from groundedness to hallucination. For our enterprise use cases, we don't want creativity, we want it to do what it has to do.

"The winners in startupland are those who can switch models faster than anyone else" -- Do you agree?
- At this point in time, probably yes; we'll see a bunch of models coming out. If you can have a LM-agnostic company that relies on LMs, that would give you a competitive advantage, but it incurs some risk of relying on others' LMs

Will there be more startup LM companies?
- Yes
- This world will change the world, and there's a giant market for people to do all kinds of interesting things.
- Many people don't need general models that know about quantum mechanics and philosophy; they jus want things that solve their specialized problems.

We've seen size of model matter less and less, it would seem -- how do you think about size of models, and does it matter as much as it used to?
- ==Data size matters more than model size==
- The LLaMA paper shows this brilliantly; training a smaller model on more data for longer (beyond Chinchilla optimality) gives you a better model.
	- It seems that data is actually much more important than model size, in terms of what's optimal.
- SamA thought that models would actually *stop growing in size* (maybe that GPT-4 hit a ceiling?)

If data is more important than model size, what does that mean in terms of who's most advantaged?
- It might mean that smaller companies are more advantages (relative to now). The secret sauce to some models (eg GPT-4) is because they went through an enormous, onerous, and EXPENSIVE project of creating data
	- Whisper project of transcribing ~every podcast in the world
	- Human preference information
	- Human expert information

So how important is proprietary data? We VCs think it's quite important...
- It's quite important
- But these very large models are actually much more sample-efficient than smaller models, including impressive in-context learning abilities
- So you need a lot of data to build new impresive models, but if you want to get started as a company solving problems in a space, you need very little data to get started.
- People often also use GPT-4 to generate data, and then they're training on that data with cheaper models -- so perhaps GPT-4 will disrupt mechanical turk!

What is pretrain(ing) data, and how does it change 
- Maybe it's useful to go through the steps of what you need to build a foudnation model
	- Need a core, pretrained model
		- This is trained on the Web; the task is just next-word prediction
	- Do supervised finetuning -- [[Instruction-Tuning]]
		- The model doesn't yet know how to follow instructions -- just continue the next words. Supervised finetuning on proprietary data is a way to get a better model.
	- RLHF or other alignment technology
		- Follow instructions in a manner that maximizes human preference reward
		- It's easier to say "X is better than Y" than it is to say "You did ABC wrong in your response"

Which company do you think has the best data acquisition flywheel?
- OpenAI.
- "OpenAI has no moat" memo -- Douwe said "This person has no idea what they're talking about"
- These places have a giant moat because it's all about Data, and OpenAI has a very deep understanding of ...
- Open source doesn't really have a chance in this regard


Evaluation?
- We aren't very good at this yet, and it's incredibly important
- Chances for some companies to become the "Moody's" etc
- GPT-3 looks like an amazing coder, but it might actually be trained on the data that it's evaluated on, which means that it's not really a good coder

Dynabench
- The idea is that you can't have a static test set, because people will overfit on that
- Ideally you want to see how easy it is for an adversarial person to mess with your model
- It used to be very easy to come up with adversarial attacks, and it's getting harder over time.

So are we going to have a next-gen set of cybersecurity companies? Focused around LLMs?
- Yes
- I also don't think that foundation model companies will (only) build this in-house

How do you think OpenAI discusses data contamination?
- They're investing more in evaluation, I think.  
- They have an army of human annotators checking their models, so they probably have a good sense of how accurate their models are, but thy might not share that with the world.

How do you think of Open/Closed?
- Frontier models are much better and expensive than everything else
- At the bottom are smaller more specialized open models
One model won't win everything -- different ones will be used for different tasks -- you might not need generality.

Open source models won't move up the pyramid though -- they're too expensive. The current models just benefit from the generosity of Meta through their release of Meta.






