https://www.youtube.com/watch?v=ayzOzZGHZy4&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=2

![[Pasted image 20240625203625.png]]

-----


![[Pasted image 20240625174745.png]]
Neural Networks can learn *total randomness* -- if you give it really bad data, it will produce exactly what it learns, even if that's really wrong.
- So the article on the left should really be: "When algorithms are trained with erroneous (or not enough) data"

Traditional ML is very model-centric.
- When we learn ML in school, a dataset is given to us, usually fairly clean and well-curated. The goal is to produce a best model for this dataset.

In the the real world, the dataset is not fixed. It comes from noisy sources, and may be able to choose to get more or less.
- The Company doesn't care what clever ML tricks you use to produce accurate predictions on *highly curated data*
- Real world tends to be messy; how do we systematically improve our data?

Fun fact: Ten of the most-used ML test sets have pervasive label errors!
- See [http://labelerrors.com](http://labelerrors.com)


What is ==Data-Centric AI==?
Often takes one of two forms:
- AI algorithms that understand data and use that information to improve models.
	- eg [[Curriculum Learning]]: The goal being to train on "easy data" first (Bengio, 2009)
- AI algorithms that *modify data* to improve AI models
	- eg Confident Learning: Removing wrongly-labeled data prior to training, so you just train on correctly-labeled stuff. If your teacher is making mistakes 30% of the time, and you compare it to a teacher who told you right things, which do you thin you'll learn better from?

In Model-Centric AI:
- Given a dataset, improve the model.
In Data-Centric AI:
- Given a model, improve the dataset.

What is *not* data-centric AI?
1. Hand-picking a bunch of data points that you think will improve a model
	- Not scalable, systematic
2. Doubling the size of your dataset to train and improve a model on more data
	- This is just classical ML! All of the work is in the model, you're just paying more money for more data.

Data-centric AI corollaries
1. ==[[Coreset Selection]]==
	- We have a dataset, and we train a model on that dataset and get 98% accuracy, but the dataset is 100B points! Can we find a 1M point subset that gets us 95% accuracy?
2. [[Data Augmentation]]
	- We can turn one image into 10 imagines by skewing, shifting, rotating, adding noise, etc. 
	- For text data, [[Back-Translation]]! 
		- "Hi, my name is Curtis" -> "Hola, me llamo Curtis" -> "Hi, I'm Chris"
			- Cool, another "version" of the same thing.
	- Note that if your initial label is wrong, then your augmentation will just multiply the problem! You want to assure quality before doing augmentation.

What are some examples of Data-Centric AI?
- [[Outlier Detection]] and Removal (handling abnormal examples in dataset)
- ==Error Detection and Correction== (handling incorrect values/labels in dataset)
- [[Data Augmentation]] (adding examples to data to encode prior knowledge)
- [[Feature Engineering]] and Selection (manipulating how data is represented)
- ==Establishing Consensus Labels== (determining the *true* label from crowdsourced annotations)
- [[Active Learning]] (selecting the most informative data to label next; minimizes the amount of data we need to collect)
- [[Curriculum Learning]] (ordering the examples in a dataset from easiest to hardest)

![[Pasted image 20240625202011.png]]
A picture from the speaker's ~2016 (?) internship at FAIR; we've been training models on MNIST for >20 years, and people assume it has perfect labels, since it's such an easy dataset. It's very high quality, but Jeff Hinton was presenting Capsule Networks... and the highlight of the talk was showing that this 5 image was labeled a 3, in the MNIST dataset!
- Years later now, we can systematically fine millions of errors!


ChatGPT was improved over GPT-3 by *improving data quality!*
- They talked to humans to produce gold data, did SFT.
- They had the model generate, and rank data based on the quality of the generation.

The Tesla Data Engine 
![[Pasted image 20240625202438.png|400]]
You have a data source, and you notice a problem -- hey, the car's in a tunnel, and we don't have a lot of tunnel data, and the car's doing weird things. So they'd collect a bunch of tunnel experiences, label them, and retrain the model, repeating the process.
The goal is to automate this process.
![[Pasted image 20240625202538.png]]
These are literally all examples of Traffic Lights! 😱
How do we possibly build a car that can navigate every traffic light in the world, and be robust to things that *aren't* traffic lights?
- How do we find systematic ways of improving our data?
![[Pasted image 20240625202636.png|400]]
Turns out that it's all about data! It's a big shocks to academics making the jump to industry.


![[Pasted image 20240625202733.png]]
For this particular task on this particular dataset, all of these data-centric methods beat all of these model-centric models. There's *something here* 
- The model-based techniques are modifying the model or loss function to not train as much on what they think is bad data.
- The data-centric techniques are actually modifying/changing the dataset.

*Before there was data-centric AI...*
- We relied mostly on human-powered solutions to improve dataset quality
	- Spend more % for higher-quality data, or more labels.
- Build custom tools to evaluate specific data (eg Tesla's data quality platform)
- Fixing data inside a Jupyter notebook, manually.
	- Maybe sorting data by the loss function, and manually examining it.

==We're going to look at ways to systematize these approaches so that they're more accurate and reliable.==

![[Pasted image 20240625203053.png]]


