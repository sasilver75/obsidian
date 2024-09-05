
# Logistic Regression

Logistic Regression is a binary classification algorithm. 

Let's say that we're trying to determine whether an image contains a cat or not.
- Let's say that a label of 1 denotes a cat and 0 denotes no cat.
- The images will be 64x64, and they're 3-channel color images of RGB, so we have a 64x64x3 matrix.

What we do is *unroll* these pixel intensity values into a single one-dimensional feature vector of length 12288! We've now got a long feature vector.

Notation:
- A single training example will be a pair of (x,y), where x is an N-dimensional feature vector, where y is going to be a label in {0,1}
- We have m training examples {(x[1], y[1]), (x[2], y[2]), ...}
	- We sometimes say that we have m[train] and m[test] examples.
- Let's define a feature matrix X and define it by taking our inputs (x[1], x[2], x[3]) and putting them so that we have our 
	- ![[Pasted image 20231213173508.png]]
	- In other courses, X might actually be created by having the x[n] examples be stacked in *rows*, (and it doesn't matter,) but it actually turns out that implementation will be easier if we do it Andrew's way!
	- This is an (n[x], m) dimensional matrix.
- What about the output labels Y?
	- Similarly, we'll stack our output labels in columns, but it's just a (1, m) dimensional matrix!![[Pasted image 20231213173654.png]]
- So for both X and Y, we've stacked x and y in columns.


Given x, we want yhat (the predicted y) to be the probability that y=1 | x

Given the parameters x and b, how do we get yhat?
Idea: yhat = w[T]x + b

But this doesn't constraint our outputs to within 0 or 1. The above is sort of what we'd do in linear regression.

So let's do instead:

yhat = sigmoid(w[T]x + b)

So we push the linear function through a sigmoid function, which looks like this:

![[Pasted image 20231213180344.png]]

Sigmoid(z) = 1 / (1+e^-z)
- If z is large, then e(z) = 1 / (1 + 0) = ~1
- If z is small, then e(z) = 1 / (1  + huge) = ~0

So when you implement logistic regressoin, we want to:
- Learn values for w and b (our parameters) such that our yhat becomes a good estimate of the probability that y=1 | x for any x in X.

When we program NNs, we'll usually keep the parameters w and b separate. In some conventions, you define an extra feature where x[0] = 1 so that x is in the space of n[x] +1 rather than n[x]. In that world:
![[Pasted image 20231213180729.png]]
- It turns out that when you actually implement a NN, it's easier to just keep b and w as separate parameters, so in this class we WILL NOT USE THIS NOTATIONAL CONVENTION!

Let's look at the cost function of logistic regression

Recall that the logistic regression equation is

yhat = sigmoid(W[T]x + b)
Where sigmoid(z) = 1 / (1+e^(-z))

So... given a set of labeled data {(x[1], y[1]), (x[2], y[2])}
How do we set w and b such we can hopefully get something like yhat[i] = y[i] out the other side (and not overfit on the data, but that comes later)

What should we use as our loss function? This is the function that we'll use to define how good our yhat is when we predict one! We'll use it to guide our parameters as well.
- When you come to learn the parameters

The loss function actually want to minimize is:
```
L(yhat, y) = -1ylog(yhat) + (1-y)log(1-yhat)

Why does this make sense?
If y = 1, then L(yhat, y) = -log(yhat)
	-> So if y=1, we want yhat to be BIG as possible to minimize this loss
If y = 0, then L(yhat, y) = -log(1-yhat) 
	-> So if y=0, we want yhat to be SMALL as possible to minimize this loss
```

While the ==Loss Function==  measures how good you're doing on a single record
The ==Cost Function== represents how good you're doing on your entire etraining set:

```
J(w,b) = 1/m SUM( L(yhat, y) )
	   = -1/m SUM( ylog(yhat) + (1-y)log(1-yhat) )
```

So we want to find parameters w and b that minimize our cost function.

Logistic regression can actually be viewed as a very small neural network!

### Gradient Descent

Recap:
![[Pasted image 20231213184400.png]]
The cost function estimates how well your W and b are doign on the training set
- Select the w and b that minimize J(w,b)

Let's say J(w,b) looks like this:
![[Pasted image 20231213184425.png]]
- So the cost function of J(w,b) can be viewed as some sort of surface. The height of the surface at any w, b determines the cost function of that combination.
- It turns out that this is a convex function (a single big bowl), so there's going to be a single minima that also happens to be the minimum. Convex functions are much easier to optimize than nonconvex ones.
	- Actually, the fact that the cost function above is convex is because of the loss function that we chose! Some loss functions are not convex :( 

So how do we get from SOME w,b to the OPTIMAL w,b?
- We start at some point on the surface by selecting a w,b and we "roll downhill!" There are various ways of doing them, but ==gradient descent== has us take the derivative of the loss with respect to each of our parameters, to give us a gradient! This gradient can be visualized as an arrow that points "uphill" from our point on the surface. We then step in the opposite direction by SOME amount (influenced by a ==learning rate== hyperparameter)
![[Pasted image 20231213184845.png]]
Because we have more than a single parameter, we're actually getting the "partial derivative" of the loss function with respect to either w and b, so we actually write is as this weird d:
![[Pasted image 20231213185243.png]]


#### Computation Graph

Let's use a simple example to explain this

J(a,b,c) = 3(a + bc)

This function has three steps:
- What's b*c? 
	- Let's say u=bc
- What's a + u?
	- Let's say v = a+u
- What's 3v?
	- This is J!

![[Pasted image 20231213191102.png]]

We can take these three steps and draw them as a computation graph as follows:
![[Pasted image 20231213191209.png]]
This comes in handy when there's some output variable (J in this case) that we want to optimize. In Logistic regression, this is going to be our loss function.

We'll see that in order to compute our derivatives, we'll go in the right-to-left direction to actually compute the derivatives!


- If J=3v, we know that dJ/dV is going to be 3
- So what's dJ/dv and dJ/da, in the previous cell?
	- See that we're still doing dJ in the numerator!
	- The answer is 3.
	- Think: If we increase (eg) a by 3, then v increases by 3. And because J is defined by 3v, J increases by 3.
- We can see that dJ/da = dJ/dv * dv/da
	- This is both the ==chain rule== and if you actually write it out, it makes sense.
	- Indeed dv/da is going to be 1, and we know that DJ/dv is going to be 3, so dJ/da is 3 also.

![[Pasted image 20231213192017.png]]
(right to left)

Here's our cleaned up graph:
![[Pasted image 20231213192718.png]]


#### Gradient Descent for Logistic Regression

Recall:
```
z = w[T]x + b
yhat = a = sigma(z)
L(a,y) = -(ylog(a) + 1-y(log(1-a)))
```
![[Pasted image 20231213194247.png]]
We want to ultimately compute derivative of the loss with respect to each of the left things.
- Let's compute dL/da first
	- dL/da = (-y/a) + ((1-y)/(1-a))
- dL/dz = dL(a,y)/dz
	- = a - y
	- It turns out that dL/dz can be expressed as dL/dz * da/dz
		- Where da/dz is a(1-a), and we've already solved for dL/da, so we just take those things and multiply them out
- Now that we have dL/dz, can we get the left things?

dL/dw[1] = x[1]]dz
dL/dw[2] = x[2]dz
db = dz

Then for each we do something like
w[1] := w[1] - alpha(dw[1])
w[2] := ...

So we don't actually want to do this in a loop for every single one of our parameters, though!

#### Gradient Descent on M examples (rather than just one)

- Let's remind ourselves of the definition of the COST function
```
J(w,b) = 1/m * SUM( Loss(a[i], y) )

# Notation
a[i] = yhat[i] = sigma(z[i]) = sigma(w[T]x[i] + b)

# It's true that the dJ/dw is the average of the dLoss/dw
dJ/dw = 1/m * SUM (dL\dw[i])

Let's initialize at 
J=0, dw[1]=0, dw[2]=0, db=0

For i=1 to m
	z[i] = w[T]x[i] + b
	a[i] = sigma(z[i])

	#compute loss
	J += -( y[i]log(a[i]) + (1-y[i])(log(1-a[i]) )

	# compute and accumulate partial derivatives
	dz[i] = a[i] - y[i]
	dw[1] += x[i] ...
	dw[2] += ...
	db +=  ...

J /= m
dw[1] /= m
dw[2] /= m
db /= m
```

There are two weaknesses with the calculation above:
- You need to write two for loops!
	- For every record in X
	- For every feature (dw1, dw2, ...)

Being able to write this with for loops help you scale to much larger datasets. ==Vectorization== helps us get past this limitation of explicit for loops!
- This is important for large datasets, where deep learning shines!

![[Pasted image 20231213201659.png]]
- Sam: I think in this case we're using the dot because we've got two one-dimensional vectors... so dotting them is just the element-wise product. I think if we had matrices of these things, we'd be doing the matmul thing.
These are parallely executed using SIMD instructions, which are single-instruction multiple data. 

What is vectorization?
- We needed to compute z = w[T]+ b
	- Where w and x are both vectors
- To compute w[T]x, if we had a non-vectorized implementation, we'd do:

```
z = 0
for i in range(n-x):
	z += w[i] * x[i]
z += b
```
- Whereas in numpy we just compute this directly in a vectorized way:
```
# Where np.dot(w,x) is w[T]x
z = np.dot(w, x) + b
```













