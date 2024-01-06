https://lilianweng.github.io/posts/2018-06-24-attention/
---

Attention is to some extent motivated by how we pay visual attention to different regions of an image, or correlate words in a sentence.

![[Pasted image 20240105231918.png]]

Human visual attention allows us to focus on a certain region with "high resolution" (ie the foveal view) while perceiving the surrounding image in "low resolution."

Given a small patch of an image, pixels in the rest provide clues about what should be displayed there! We *expect* to see a pointy ear in the yellow box, because we've seen a dog's nose, another pointy ear on the right, and Shiba's mystery eyes... However, the sweater and blanket at the bottom wouldn't be as helpful as those doggy features.

Similarly, we can explain the relationship between words in one sentence or close context.
![[Pasted image 20240105232351.png]]
- When we see "eating" above, we expect to encounter a food word very soon.
- In a nutshell, attention in deep learning can be broadly interpreted as a *vector of importance weights.*
	- In order to predict or infer an element (like a pixel in an image or a word in as sentence), we estimate (using an attention vector) how strongly that element relates ("attends to") other elements in the sequence. We then take the sum of those other elements' values weighted by the attention vector as an approximation of the target.

## What's wrong with Seq2Seq model?
- The ==seq2seq== (Sutskever et al., 2014) was born in the field of language modeling; it aims to transform an input sequence (source) into a new one (target) and both sequences can be of arbitrary length.
	- Examples of transformation tasks include machine translations between multiple languages in either text or audio, question-answer dialog generation, or even parsing sentences into grammar trees.
- The seq2seq model normally has an encoder-decoder architecture, information of:
	- An `encoder` processes the input sequence and compresses that information into a context vector of a fixed length. This representation is expected to be a good summary of the *meaning* of the whole source sequence -- it's the *gestalt* of the sequence!
	- A `decoder` is initializer with the context vector, and aims to emit the transformed output. The early work only used the last state of the encoder network as the decoder initial state.
	- Both the `encoder` and the `decoder` are [[Recurrent Neural Networks (RNN)]], using either [[LSTM]] or [[GRU]]
































