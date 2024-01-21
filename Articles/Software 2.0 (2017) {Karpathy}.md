Link: https://karpathy.medium.com/software-2-0-a64152b37c35

Sometimes people refer to NNs as "just another tool in the ML toolbox," having pros and cons -- they work here and there, and sometimes you use them to win Kaggle competitions.
- This misses the forest for the trees -- NNs are not just another classifier; they represent a fundamental shift in how we develop software -- ==Software 2.0==!

The "classical stack" of ==Software 1.0== is what we're all familiar with
- Python, C++, etc.
- Consists of *explicit instructions* to the computer written by a *human programmer*
- By writing each line of code, the programmer identifies a specific point in program space with some desirable behavior.

In contrast, ==Software 2.0== is written in a much more abstract, human-unfriendly language, such as the weights of a neural network, such as the weights of a neural network.
- No human directly writes this code
- Instead, our goal is to specify some goal on the behavior of a desirable program.
	- "win a game of go"
	- "Satisfy a dataset of input:output pairs of examples"
- ==We write a rough skeleton of the code that identifies a subset of program space to search, and then use computers to *search this space* for a program that works!==
	- In the context of Neural Nets, we restrict the search to a continuous subset of program space where the search process is guided (and made more efficient) by backpropagation and stochastic gradient descent.

![[Pasted image 20240120171940.png]]
In Software 1.0:
- Human-engineered source code (eg some .cpp files) are compiled into a binary that does useful work.

In Software 2.0:
- The source code comprises:
	- The dataset that defines the desirable behavior
	- The neural network architecture that gives the rough skeleton of the code, but leaves many details (weights) to be filled in.
- The process of *training* the NN *COMPILES* the dataset *INTO* the binary (the final, trained neural network).
- In most practical applications today, the neural net architectures and training systems are increasingly standardized into a **commodity**, so ==most of the active "software development" takes the form of curating, growing, massaging, and cleaning labeled datasets.==


This alters the programming paradigm, and ==suggests a new paradigm== where:
- Software 2.0 Programmers (data labelers) edit and grow the datasets
- A few 1.0 Programmers maintain and iterate on the surrounding training code infrastructure, analytics, visualizations, and labeling interfaces.


It turns out that  a large portion of real-world problems have the property that it is significantly easier to collect the data (and to identify a desirable behavior) than it is to explicitly write the program.
- "I'll know it when I see it."
- "I can say 'X is better than Y, as a solution to Z' but I can't easily hand-write a solution to Z myself."

Because of this, we're witnessing a massive transition across the industry, where a lot of 1.0 code is being ported to 2.0 code.
- ==Software (1.0) is eating the world, and now AI (Software 2.0) is eating Software==.


---
### Ongoing Transition
- Let's briefly examine some concrete examples:
	- ==Visual Recognition== is used to consist of engineered features with a bit of machine learning sprinkled on top at the end (e.g., an SVM). Since then, we discovered much more powerful visual features by obtaining large datasets and then searching in the space of CNN architectures, which found the features for us.
		- More recently, we don't even trust ourselves to hand-code the architectures, and we've begun *searching over architectures as well!*
	- ==Speech Recognition== used to involve a lot of preprocessing, gaussian mixture models, and hidden markov models... But today, they consist almost entirely of neural net stuff! 
		- Fred Jelinek (1985): "Every time I fire a linguist, the performance of our speech recognition system goes up."
	- ==Speech Synthesis== has historically been approached with various stitching mechanisms, but todays' SotA models use large ConvNets (eg WaveNet) that produce raw audio signal outputs.
	- ==Machine Translation== has usually been approached with phrase-based statistical techniques, but NNs are quickly becoming dominant. 
	- Games: Explicitly hand-coded Go and Chess programs have been the way to do it, but AlphaGo Zero has now become the strongest game player.
	- Databases: More traditional systems are also seeing the early hints of a transition; Learned index structures in databases replace core components of DBMSs with NNs.

----

## The Benefits of Software 2.0
- Why should we prefer to port over complex programs into Software 2.0?
- One easy answer should be that they work better in practice, but there should be some other reasons:
	- ==Computationally homogenous==
		- A typical NN is made up of a sandwich of only two operations: matrix multiplication and thresholding at zero (ReLU or similar activation functions). 
		- Compare this to the instruction set of classical software, which is much more complex. {This uhhh hides a lot of truth}
	- ==Simple to bake into silicon==
		- As a result of the above, since the instruction set of a neural network is relatively small, it's significantly easier to implement these networks much closer to silicon, using (eg) ASIC chips, and so on.
	- ==Constant running time==
		- Every iteration of a typical NN forward pass takes exactly the same number of flops; there's zero variability based on the diffrent execution paths that your code could take through a sprawling C++ code bae.
			- {The upside is that a very difficult LLM-posed question takes N flops, just as a very easy LLM-posed question takes N flops. The downside is the same thing, viewed in reverse.}
	- ==Constant memory use==
		- No dynamically allocated memory anywhere, so little possibility of swapping to disk, or memory leaks that you have to hunt down in your code.
	- ==It's highly portable==
		- A sequence of matrix multiplies is easy to run on arbitrary computational configurations compared to classical binaries or scripts that compile down to a specific target.
	- ==It's very agile==
		- If you had C++ code that you wanted to make twice as fast, it would be highly non-trivial to tune the system for a new spec.
		- In Software 2.0, we can take our network, remove half the channels, retrain, and there -- it runs at exactly twice the speed and works a little worse!
			- Conversely, if you happen to get more data/compute, you can just make your program work better by adding more channels and retraining.
	- ==Modules can meld into an optimal whole==
		- Our software is often decomposed into modules that communicate through public functions, APIs, or endpoints; However, if two Software 2.0 modules that were originally trained *separately* interact, we can easily backpropagate *through the whole*!
	- ==It's better than you!==
		- Finally, a NN NN is a better piece of code than either you or I can come up within a large fraction of valuable verticals (anything doing with language, images/video, sound/speech).

---
## The limitations of Software 2.0
- The 2.0 stack has some disadvantages too!
	- Can fail is unintuitive and embarrassing ways
	- Silently fail by adopting biases in training data that don't exist at production
	- Adversarial exampless and various NN-specific attacks

-----
## Programming in the 2.0 stack
- Software 1.0 is code we write; Software 2.0 is code written by the optimization based on an evaluation criterion (like "classify this training data correctly"). 
	- It's likely that any setting where the program isn't obvious but one can repeatedly *evaluate the performance of it* (e.g. - did you classify some images correctly?) will be subject to this transition.
- There's still a lot of work to do; We've built up a vast amount of toolign that assists humans in writing 1.0 code, like powerful IDEs with features like syntax highlighting, debuggers, profilers, go to def, git integrations, etc. In the 2.0 stack, the programming is done by accumulating, massaging, and cleaning datasets.
	- For example, when the NN fails in hard or rare cases, we don't fix those predictions by writing code, but by including more labeled examples of those cases.
	- Is there a Github 2.0 for Software 2.0?
	- What's the Conda equivalent for NNs?

---















