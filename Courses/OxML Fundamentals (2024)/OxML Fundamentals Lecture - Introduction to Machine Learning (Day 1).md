Speaker: Volodymr Kuleshov (Who also teaches [good courses](https://www.youtube.com/user/vkuleshov/videos) on Applied Machine Learning and Deep Generative Models at Cornell Tech)

600 participants from over 87 countries, making this one of our most diverse cohorts yet. This is the 5th OxML summer school; Trained 4,500 individuals in ML.

---

The content of this is inspired by this Applied Machine Learning course that he teaches at Cornell Tech, which are freely available on Youtube.

## Part 1: Intro

ML in Everyday life:
- Search Engines (eg Google)
- Personal Assistants (eg Siri, OpenAI, Gemeni)
- Self Driving cars

What makes these three examples instances of ML instead of regular programming? Let's look at the self-driving car example

How might we try to build an object-detection system without using ML (for self driving)?
- We might try to write some rules for identifying cars by whether it has 4 wheels or not, etc. We try to do some top-down rules that will ultimately fail.
- It's better to have an algorithm that *learns* these rules (specifically, the ones that help us maximize performance on an objective function). 

Certain types of computer programs can really *only be written* using machine learning! 

There are many applications of [[Supervised Learning]]
- Classifying medical images
- Translating between languages
- Detecting objects in an autonomous driving scenario

## Part 2: Ordinary Least Squares

Let's get a precise definition of one ML algorithm called ***Ordinary Least Squares***, which is very simple, but very important and powerful.

Components of a supervised learning problem

$Dataset + Learning Algorithm \rightarrow Predictive Model$ 
- Dataset contains (Inputs, Targets)
- Learning Algorithm consists of: (Model Class, Objective, Optimizer)

The output of this process is a predictive model that's able to map inputs to targets. The hope is that this performance generalizes from inputs in the training dataset to inputs that we'll see at inference time in the wild.

![[Pasted image 20240519160256.png]]
Above
- We define our dataset as a matrix $X$ , where each row in the matrix is an input, with the inputs having $d$-dimensionality.

We can similarly define our Target from our (Input, Target) as a vector, of the form:
![[Pasted image 20240519160702.png]]
Above:
- If we're predicting diabetes risk, $y^{(1)}$ is the known diabetes risk of patient 1, corresponding to the first row in our $X$ matrix from earlier.

Now let's talk about the model family that we'll use to make predictions about diabetes risk for *new patients*:
![[Pasted image 20240519160826.png]]
Above:
- We don't know the true mapping between (Input, Target), but we have to choose something; Here, we're assuming that the true label for $y$ can be well-approximated by some model $f$ of the input data $x$ , and we're assuming that $f$ is a linear function of $x$ and our parameters $\theta$ .

We want to determine the best *family of models* that describes the data that we have, and for a given model family, we want to have a way to learn the model parameters $\theta$ that maximize the performance of the model.

To do all of this, we need a way to compare model/parameter combinations with eachother. We do this with an objective function:

![[Pasted image 20240519161124.png]]
Here, we're going to look at the sum of squared errors of our predicted model, where an error is the difference between the true label and the label predicted by our model.

We can apply some algebra to rewrite this objective function $J(\theta)$ into a vectorized form:
- Note that we can write this summation into a dot product of two vectors, each being $y-X\theta$ ... If we take the matrix vector product between our parameters $\theta$ and our data matrix $X$, the result of the operation is going to be a vector, where the i'th entry of theta is $\hat{y}^i$ is the prediction on the i'th patient. When we do $y - {that vector}$ , we get a vector where each element is the error of the prediction. 
	- Again, $y-X\theta$ is a vector in n dimensions, where the i'th dimension corresponds to the difference of $y^i - \theta^Tx^i$  from the original function.

![[Pasted image 20240519161349.png]]
Also note that  if we take the dot product of this (y-Xtheta) vector with itself, we'll get back by definition the summation of the vector. That's also the squared euclidean norm, which is the other notation that we're going to take.
- We can equivalently rewrite J(theta) into a form that only requires y, X, theta.

Why is this interesting?

Now that we have this simple objective, we can optimize it using a little bit of Calculus. We use the notion of Gradients
![[Pasted image 20240519161838.png]]
Recall from Calculus 1, where a univariate funciton's derivative being 0 denotes a minimum or maximum. Similarly, a Gradient defines a derivative to multivariate functions... We define this gradient as being a vector of all partial derivatives of our objective function, with respect to theta.

Then we can compute the gradient of the mean squared error from the previous slide:

![[Pasted image 20240519161950.png]]

... more math omitted ...

Model Family:
- We assumed that the model that we fit (a linear model) was a good fit for our data, but maybe another model family (eg polynomial) might be better!

![[Pasted image 20240519163219.png]]

One way in which we could do this is by defining a feature vector $\phi(x)$ which takes an input x, and derives *features* based on x , crafting a more interesting form of x that we can present to the model.

Now consider fitting a linear model not over x directly, but over the *features* of x, $\phi(x)$ !
![[Pasted image 20240519163347.png]]
Simply by definition of $\phi(x)$

- Now this model is non-linear in the input variable x, meaning that we can model complex data relationships.
- But it's still a linear model as a function of the parameters $\theta$, meaning that we can use our familiar ordinary least squares algorithm to learn these features!

![[Pasted image 20240519163650.png]]
Above: Example of us doing some feature engineering; this is our new data matrix $X$, where patients (rows) are described with three features and a vector of ones.
- Now we can fit a linear model to this dataset; we can apply the same OLS model to this matrix.

![[Pasted image 20240519163818.png]]
So the line is nonlinear, because our features are nonlinear :)
Seems to fit our data a little better (visually) than our simple linear model. It's linear in the parameters $\theta$ and $\phi$ , but not linear in the inputs $x$. This means that it's more expressive in a model that's only linear in $x$, but because the it's linear in the parameters, we can still apply our OLS algorithm by giving it nonlinear *features*.

So it seem like higher-degree polynomials must be better, right?

![[Pasted image 20240519164526.png]]

Not so! See that our model is in some sense good, in that it passes through every datapoint that we have... the training data error is likely going to be very good -- but it's unlikely that this model is going to generalize to other data out-of-sample.

This is [[Overfitting]], which is one of the most important problems to know about when applying Machine Learning.
- A very expressive model (eg a high-degree polynomial) fits the training dataset perfectly
- The model makes highly-incorrect predictions outside of its dataset, and doesn't generalize.
- (We make superfluous "errors due to variance")

Solutions: [[Early Stopping]], [[Regularization]], Collecting more training data, Using a simpler model family

The opposite of overfitting is [[Underfitting]], in which our data is complicated, but we try to fit (eg) a linear model to it -- our model doesn't have the capacity to fit to the data, and doesn't even perform well during training.
- We need to make our model more expressive/rich by:
	- Adding more interesting features to our model ([[Feature Engineering]])

The challenge is to strike a balance between underfitting and overfitting.


















