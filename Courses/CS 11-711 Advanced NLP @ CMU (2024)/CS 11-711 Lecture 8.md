# Topic: Fine-tuning and Instruction Tuning
https://www.youtube.com/watch?v=KLJ3EEo8aPU&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=8

---

Full Fine-tuning
- We simply continue training our model on whatever task that we want
- This can take lots of memory, and can even be relatively unstable to other alternatives!
	- For example, a 65B model (largest LLaMA 1) with 6-bit precision takes much more memory than you'd expect!
![[Pasted image 20240615171306.png]]
Forward/Backward pass depend on how many tokens you have in your sequence, etc.
Overall, this takes ~1000GB of memory, which isn't great!

Luckily, there have been some advances since then.

![[Pasted image 20240615171533.png]]
See that our best available GPUs right now are 80GB; the B200 GPU will pack up to 192GB, a 2.4x increase over H100s.
So it goes without saying that you've got yourself a distributed systems problem, even for finetuning a relatively small 70B model.

So how can we overcome this limitation of GPU memory?

MultiGPU Training!
- We can just throw more hardware at the models, and distribute the models over multiple GPUs.
- The most canonical version of this is called DeepSpeed Zero
	- Works by partitioning optimization over different devices.
	- There are different stages of DeepSpeed Zero;
		- The first is this Baseline Row; we hold all of these on each GPU
		- The second is that we partition the optimizer state across GPUs. Because the op
