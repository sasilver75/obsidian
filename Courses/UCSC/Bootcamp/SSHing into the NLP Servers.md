
SSH into nlp-gpu-01-be-ucsc.edu

ssh <you_username>@nlp-gpu-01.be.ucsc.edu

Enter your GOLD password


More instructions here:
https://gist.github.com/kingb12/ddabe0f9d7d2427dec90931dbca5d778
To set up a SSH Config (a template to do remember settings and login and such, so you can just login with `ssh nlp-gpu-01`)
SSH keys will help you safely authenticate without typing password

Let's say you end up on the GPU machine and you want to know some things about what's going on on the server?
- `nvidia-smi`: Shows you how much GPU memory is being used; defines whether you can fit your model/program/data onto the GPU; how much GPU utilization is occurring.
	- If the utilization is high, both your and the existing job will work, but both will be slower.
	- The utilization is pretty spiky, so it's nice to have it refresh.
If you want it to refresh:
`watch -n .1 nvidia-smi`

How do I find them?
Note the PID (eg 9389)

Use ps to find the owner
ps -o pid -o user -p 9389
- -o: tell ps what to output (PID and User)
- -p: Tell ps what PID to do this for

To check what your current running process is?
```
ps -o pid -o user -o comm, args=ARGS -o etime | grep your_username
```

How to pick your GPUs
- There are 6 GPUs, and the general , and the general way to do this is with an environment variable called CUDA_VISIBLE_DEVICES
	- Pytorch/and NN library will honor this environment variable and only consider those GPUs as visible.
	- So you can export CUDA_VISIBLE_DEVICES=4

If you do 4,5  for instance then you can use multiple GPUs

If you don't set this environment variable, it's possible at the NN library you're using will do something weird, like select the first GPU, or use all of the GPUs, etc. Most of them will imagine that you're the single tenant of the server

Nilay recommends Anaconda (more specifically, miniconda) to manage both python package and non-package packages (since you won't have admin rights on the gpu box that you ssh into)
- Otherwise, he says that using pyenv is also fine.
- Though you can do things like modify the cuda version, etc with conda.


Instructions in the slides about how to creatre a "Screen" which is like an ssh connection of sorts, except it won't terminate your jupyter notebook when you leave.

He usually
- Runs thelong  ssh -L 21042:... command
- Runs the ssh command
- Runs the jupyter notebook --port command