---
tags:
  - paper
---

Link: https://arxiv.org/abs/2212.10560
Date: December 2022
Authors: Yizhong Wang

@EugeneYan: "Followed by the new Back-Translation paper" (https://arxiv.org/abs/2308.06259)

-------


### Latent Space Paper Club Notes (January 24, 2024)
- Voice Memo recorded on M1 Macbook
![[Pasted image 20240124120926.png]]
![[Pasted image 20240124121011.png]]
![[Pasted image 20240124121208.png]]
![[Pasted image 20240124121140.png]]
- Every task we have can have one or more input/output sequence.
- Given a set of instructions, tasks having >=1 outputs, we have an LLM to produce an output given the task instruction and the input.
	- Note that the additional input can be empty (see the cat example two pics up)

![[Pasted image 20240124121253.png]]
![[Pasted image 20240124121351.png]]
If it's classification they do one method, if it's not (write a letter), they use another method.
- They went pretty overboard for a basic classification task for an LLM, but this is how they chose to do it.

![[Pasted image 20240124121434.png]]
- In the paper they deal with Classification vs Instruction prompts differently
- They came up with the method of having input-first and output-first response;s in this case you generate the input first, or the output
- For clasification, they decided to generate output first, then input; they ran into issues to have a clasification class to determine if grammar was bad... the model kept generating correct-grammar inputs (lol), so they just chose to do it in reverse so that it would generate bad-grammar inputs.

![[Pasted image 20240124121544.png]]
Interesting: They went heavy on the few shot propmting!

![[Pasted image 20240124121615.png]]

![[Pasted image 20240124121626.png]]
![[Pasted image 20240124121652.png]]
- Given classification task, gnerate the input of something correspodning to a label of (eg) something that's gramatically incorrect; it will generate a gramatically incorrect sentence.


![[Pasted image 20240124121716.png]]


Filtering
![[Pasted image 20240124121734.png]]
- In the time of GPT-3 ; no multi-modal
- Removed anything talking about images/pictures/graphs; sometimes LLMs get wild, so they started removing repetition-invalid things, etc.
- Filtered for length (too long, too short).
- In order to promote the diversity of new samples, they'd only add new instructions when they had a ROUGE-L score of < .7 ; so they wouldn't add things that were too similar to what they already had.

![[Pasted image 20240124121841.png]]
Overview of the process

Question @ Eugene Cheah: Did the paper credit any prior sources?
A: ALPACA, Blaze, ... they have a section on related work where they mention a few papers working on similar things.

Results section (He thought it was out of data at this point; a lot ofit was finetuning based GPT-3; a lot of later work has taken over, so he doesn't go too into depth in the evals for this, because they're sort of outdated.)

![[Pasted image 20240124122057.png]]
- Sampled 1/200 samples generated; manually looked at it as researchers; they learned that even if there was noise (some of the samples were wrong), it still helps with guidance on instruction following. Even if the answer is incorrect, it's useful to help with following basic instructions. Even having some noise (bad examples generated) helps a little bit.
- 





