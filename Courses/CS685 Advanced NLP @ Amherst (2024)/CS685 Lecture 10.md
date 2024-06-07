
# Topic: Parameter Efficient Fine-Tuning

Discussion of [[Prompt Tuning]] (2022)
![[Pasted image 20240523221018.png]]
Blue vectors are added to the prompt, and when we finetune the model, we also finetune these vectors.

![[Pasted image 20240523221046.png]]
Note that we keep the rest of the model frozen. Think about why! Downsize of this is that dL/de1 requires backpropagating through the whole model!

---

[[Low-Rank Adaptation|LoRA]]: Low Rank Adaptation (2022)
Developed by people at Microsoft (Tim Dettmers)

Now, it's the dominant family of approach for (consumer) finetuning of LLMs

The key insight is that when you make updates to matrices, the update is generally low-rank.
- ==The rank of the matrix is basically measuring how many of the rows/columns in the matrix are linearly independent from eachother==
- This suggests all the information needed to update these models is contained in a lower-rank subspace; it's not necessary to have 1000x1000 matrices, etc.

dL/dW
- If W is an m x n matrix, then dL/dW will also be an m x n matrix
- Authors of the LoRA paper say that we don't need to use a full m x n matrix to represent the partial derivative dl/dW; instead we can represent this with a product of two lower-rank matrices that approximate the dl/dW matrix!

If W = m x n
Perhaps we approximate W with A and B
Where A = m x r
	    B = n x r

And $AB^T = W$ 

We can pick any inner dimension r

Now in LoRA:

==The equation:==
$h = f((W_{pre} + AB^T)x)$
- this basically says that $AB^T$ represents the change to our pretrained model's weight parameters, and we just add those to our "pre" W matrix.

During finetuning, we only update our A and B matrices, computing $dL/dA$ and $dL/dB$, which are both much smaller than the alternative, $dL/dW$

At the end of this LoRA finetuning process, we've got a separate A and B for each tuned weight matrix! (The paper later shows that they really only need to do it for the projection matrices Q K V in each attention head, they don't really even need to tune the FFNN layers at all)

To recover the "new" weights of the finetuned model, we can just do

$W_{New} = W_{Old} + AB^T$ 

And then later we can simply do

$f(W_{new}x)$ 

----

[[Quantized Low-Rank Adaptation|QLoRA]]: Quantized Low-Rank Adaptation

Quantization refers to trying to reduce how much memory these parameters take on the GPU

In a "normal" model, we might represent each parameter with a 32-bit floating point number (FP32), covering a huge range of possible values, but also taking a lot of space.

Maybe we can use a less-expressive representation?
- 4-bit, 8-bit integer quantization

There's some performance degradation, but less than you'd expect.
