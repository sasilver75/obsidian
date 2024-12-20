
References:
- [Jonathan Berant (Tel Aviv University / Google) / Towards Robust Language Model Post-training](https://youtu.be/2AthqCX3h8U?si=jiOI1gvV9vw99rsa&t=960)

Problem with [[Direct Preference Optimization|DPO]]
![[Pasted image 20241214235315.png]]
Probability of both preferred responses and dispreferred samples both went down!
In many cases, this leads to outputs that are catastrophic/degenerative...
It's like overfitting, but... probability mass, when you generate from the model, is concentrated on outputs that are very bad.