---
tags:
  - article
---

Link: https://timdettmers.com/2018/12/16/deep-learning-hardware-guide/

----
Deep learning is very computationally intensive, so you'll need a CPU with many ores, right? You need a fast CPU, right? Nope! Don't waste money on hardware that's unnecessary. 

Let's go over some common mistakes, ranked by severity:

### GPU
Mistakes you can make include:
1. Bad cost/performance
2. Not enough memory
3. Poor cooling

Look out for cooling
- If you want to stick GPUs into PCIe slots that are next to eachother, you should get GPUs with a blower-type fan, or you'll run into temperature issues that throttle your GPUs and make them die faster.

### RAM
Mistakes include
1. Buying RAM with *too high* a clock rate
	- RAM clock rates are marketing stints where RAM companies lure you into buying "faster" RAM which actually yields little to no performance gains.
	- It's important to know that RAM speed is pretty much irrelevant for fast CPU RAM -> GPU RAM transfers. This is because if you use pinned memory, your mini-batches will be transferred to the GPU without involvement from the CPU... Spend your money elsewhere!
2. The second mistake is to buy not enough RAM to have a smooth prototyping experience.
	- RAM size doesn't affect deep learning performance, but it might hinder you from executing your GPU code comfortably (without swapping memory to disk). You should have enough RAM to comfortably work with your GPU -- This means that you should have AT LEAST the amount of RAM that matches your biggest GPU's VRAM!
		- So if you have a Titan RTX with 24GB memory, then have at least 24GB of RAM.
	- In Kaggle competitions, having additional RAM is very useful for feature engineering, says the author. So buy a lot of cheap RAM, is what the author says!

### CPU 
- The main mistake people make is that people pay too much attention to the PCIe lanes of a CPU; you shouldn't care much about this (has no effect on deep learning performance).
	- Instead, just make sure that your CPU and motherboard combination supports the number of GPUs that you want to run. 
- The second most common mistake is to get a CPU that's too powerful!
	- The CPU does little computation when you run your deep nets on a GPU. Mostly, it initiates GPU function calls and execute CPUs functions.
	- The most useful application for your CPU is data preprocessing.

Two common data processing strategies which have different CPU needs:

```
PREPROCESSING WHILE YOU TRAIN
1. Load mini-batch
2. Preprocess mini-batch
3. Train on mini-batch

PREPROCESS BEFORE ANY TRAINING
1. Preprocess data
2. Loop:
	1. Load preprocessed mini-batch
	2. Train on mini-batch
```
For the first strategy, a good CPU with many cores can boost performance significantly! 
For the second strategy, you don't need a very good CPU.

When people think about fast CPUs, they usually think about the clock rate. 4GHz is better than 3.5GHz, or is it?
- This metric actually isn't useful for comparing across different processors, and it isn't the best measure of performance.
- Again, there's little computation to be done by the CPU in the case of deep learning.

### Hard Drive / SSD
- The hard drive is not usually a bottleneck for depe learning, but if you do stupid things, it will hurt you!
- If you read your data from disk when they're needed (blocking wait), then a 100MB/s hard drive will cause you about 185 milliseconds for an ImageNet mini-batch of size 32 -- ouch!
	- But if we asynchronously fetch the data before it's used (eg torch's vision loaders)... then it will be better :) 

- Recommends an SSD for comfort and productivity. Programs start and respond more quickly, and preprocessing large files is quite a bit faster. Buying an NVMe SSD will result in an even smoother experience when compared to a regular SSD.

### Power Supply Unit
- Generally, you want a PSU that's sufficient to accommodate all your future GPUs! GPUs typically get moer energy efficient over time, so while other components will need to be replaced, a PSU should last a long while -- it's a good investment.
- Calculate required watts by adding up the watts of your CPU and GPUs with an additional 10% of watts for other components and as a buffer for power spikes.
- Remember:
	- Even if the PSPU has the required wattage, it might not have enough PCIe 8-pin or 6-pin connectors! Make sure to have enough connectors on the PSU to support all your GPUs!

### CPU and GPU Cooling
- Cooling is important and can be a significant bottleneck, reducing performance more than poor hardware cohices do.
- You should be fine with a standard heat sink for your CPU, but for your GPU you might want to make special considerations.
- Air cooling is safe and solid for a single GPU, or with multiple GPUs with space between them (eg 2 GPUs in a 3-4 GPU case).
	- But when you have 3-4 GPUs in a case, temperature can be hard to manage. GPUs will throttle themselves if they get too hot! Typical pre-programmed schedules for fan speeds are badly designed for deep learning programs.
	- Since NVIDIA GPUs are first and foremost gaming GPUs, they're optimized for windows!

### Cases
- Go cheap here. The case should fit your GPUs but that's it. Buying large towers for the additional fans is largely irrelevant (a 2-5 degrees C decrease, not worth the investment and the case bulkiness).
- Make sure that when you select a case, it supports full-length GPUs that sit on top of your motherboard -- be suspicious if you buy a small case

So for 1 GPU, air coling is best. For multiple GPUs, you should get blower-style air cooling and accept a tiny performance penalty (10-15%), or otherwise pay extra for difficult-to-setup water cooling.


### Motherboard
- Your motherboard should have enough PCIe ports to support the number of GPUs you want to run (usually limited to 4 GPUs)
- Make sure that the motherboard actually supports the GPU setup that you want to run!


### Monitors
- Buy a few of them.














