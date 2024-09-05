Github Repo: [llamafile](https://github.com/Mozilla-Ocho/llamafile)

A Mozilla project that builds on top of the [[llama.cpp]] project.
Depending on what CPU/model/weights you're using, your CPU inference will be 30-500% faster.
One file which runs 100% locally and privately.

Inside every Llamafile is a polyglot file format that enables execution on MAC/Windows/Linux/etc CPUS. Enables running on NVIDIA and AMD GPUs too with TinyBLAS.

LLMs spend the majority of their time doing matrix multiplication; they've made it go 10x faster on CPUs.

Raspberry Pi 5: 8tok/sec -> 80tok/sec
Intel Core i9: 100tok/sec -> 400tok/sec
AMD Threadripper: 300tok/sec -> 2400tok/sec

Patience is all you need! Performance matters, but it's not often the thing we care about -- often, it's intelligence, and RAM is cheap with CPUs! It's reasonable to put 512GB of RAM in your workstation and be able to run near-frontier models.
> Tired: Running Mixtral 8x7B on two 4090s
> Wired: Running Mixtral 8x22B on a $350 Ryzen CPU

