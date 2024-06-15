References: 
- [VIDEO: CMU Advanced NLP 2024 (6): Generation Algorithms](https://youtu.be/96MMXDA7F74?si=hjzP2vmai5keSfNv&t=3749)

![[Pasted image 20240614223534.png]]
You can condition your model on a mixture of human and model-generated text; You can also have the model generate some text, then have a human modify some of it (eg to match their writing style), then have the model continue to generate text.

![[Pasted image 20240614223629.png]]
"fine-grained replacement"
- User chooses points to intervene, adds additional constraints.
- This can be done through input manipulation, where you prompt with different constraints, or you could use some sort of infilling model (eg for code), or for decoding changes like we talk about in [[Constrained Decoding|Constrained Generation]] (eg to write "more sad text", like the FUDGE example in the link)

![[Pasted image 20240614223727.png]]
Humans choosing which generation to return is common too. In ChatGPT, the human can reject the text by pressing "regenerate"; this is also a way of controlling decoding on a *human* level.

But you don't necessarily need a human! 
![[Pasted image 20240614223831.png]]
You can use tools like [[Tree of Thought]]; the idea is here is to generate several smaller sequences, and use a model to choose which one to continue.
- The idea is that, through prompting, we achieve something that (if you squint) looks a lot like [[Beam Search]], where we use a signal from an external source to control exploration of a generation space at a broader level than on single tokens.