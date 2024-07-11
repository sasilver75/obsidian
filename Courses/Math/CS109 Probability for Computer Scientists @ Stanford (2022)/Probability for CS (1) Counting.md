https://www.youtube.com/watch?v=2MuDZIAzBMY&list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg

Professor: [Chris Piech](https://stanford.edu/~cpiech/bio/index.html), Assistant Processor @ Stanford
He has a perfect [5/5 on RateMyProfessor](https://www.ratemyprofessors.com/professor/1832866), and students seem to love him.

---

One of the two fundamental rules of counting
![[Pasted image 20240703232633.png]]
A first rule to start thinking about counting (this is sort of the counting version of the [[Multiplication Rule of Probability]]).
- So if we have two dice, where one is 6 sided and the other is 8 sided... And we roll both of them (knowing that the first result doesn't effect the second result), we can just figure out the number of outcomes using multiplication: 6 x 8 = 48 possible joint outcomes.

![[Pasted image 20240703233041.png]]
For the first pixel, the color can be any of 1 of 17 million distinct colors. 
- We determine the color of the first pixel
- We determine the color of the second pixel (this is unchanged by the first one)
- ...

$17 million^{12}$ 


The second of the two fundamental rules of counting
![[Pasted image 20240703233900.png]]
(as long as there aren't elements in both sets)
![[Pasted image 20240703233915.png]]
So either the first two are fixed or the last two are fixed
The number is going to be {the number where the first two are fixed} + {the number where the last two are fixed}
If two are fixed, then there are 4 remaining, each of which has 2 possible options, so 2^4 = 16.
So we would think the answer is 16+16=32

BUT WAIT! There's some double counting going on!

So it's actually 32 - 4 = 28

---

How many unique ways are there of re-arranging the letters BOBA?
- Is it 4!? Ehhh, that would count BBOA and BBOA as different things (though they have different be orderings)

Well for the easier problem where every letter is unique: ABCD
- Step1: Choose letter, there are 4 choices to choose from
- Step2: Choose letter, there are 3 choices to choose from
- Step3: Choose letter, there are 2 choices to choose from
- Step 4: Choose letter, there are 1 choices to choose from

4!

So how do we deal with the repeated letters? next class!









