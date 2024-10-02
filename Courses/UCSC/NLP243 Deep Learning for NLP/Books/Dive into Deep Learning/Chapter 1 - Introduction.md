Link: https://d2l.ai/chapter_introduction/index.html

---

There are many problems that even elite programmers struggle to come up with solutions from scratch; reasons can vary.
- The pattern may be difficult to describe, or even understand
- The pattern may change over time

Oftentimes we're able to perform feats whose steps we can't even describe to ourselves. 
- We can instead collect a huge dataset containing examples (eg) of audio snippets and associated labels, indicating which snippets contain the wake word.
- We tune our parameter of our model such that the model is then able to predict the labels associated with the input data.

Process:
1. Start off with randomly initialized model parameters; model outputs garbage.
2. Grab some data (eg audio snippets) and predict corresponding labels.
3. Compute some loss function and update parameters to minimize the loss on that training batch.
4. Repeat (watching out for overfitting)

You can think of it as "inverse programming", or programming with data.

We need:
1. Data to learn from
2. A model of how to transform the data
3. An objective function that quantifies how well (or badly) the model is doing
4. An algorithm to adjust model's parameters to optimize objective function

 Each ==example== typically is considered as a set of ==features==, which are used to predict outputs in the model.
 - For a house, this might be square footage, color, zip code, number of bedrooms
 - For a picture, these might be the pixels in each color channel for each pixel.

Some data may be:
- High-dimensionality (Each example has many features, eg channel pixels)
- Low-dimensionality (Each example has few features, eg zillow house features)

