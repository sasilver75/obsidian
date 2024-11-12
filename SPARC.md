Sort of like CLIP, but it works on patches instead of the whole image
See: https://youtu.be/rUQUv4u7jFs?si=KVixPOLT24CdgEI4
So it uses local information instead of (?) the global information in images, for more fine-grained alignment.

![[Pasted image 20241112114827.png]]
See that previous methods like [[CLIP]] just spread the attention mask around (Yeah, the horse is in the image). Only SPARC is able to localize this. 