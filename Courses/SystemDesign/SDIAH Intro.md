System Design In a Hurry: https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction

_________________ 
# Introduction

The intent here is to teach the last and most impactful 20%, not to teach you from scratch. It won't build mastery, but it's still high-leverage. Other system design materials are LM slop or go to levels of depth that you won't likely cover in an interview.

![[Pasted image 20250510162928.png|750]]

It's recommended that you read this guide in order, skipping sections that you already know.
If you're really short on time, we recommend the ==Delivery Framework== section, skimming ==Key Technologies==, and spending any remaining time studying the ==Core Concepts== section.

![[Pasted image 20250510163040.png|450]]
Above: The four common types of software design interviews
- ==Product Design==: Often called "Product Architecture" or "System Design" interviews; the most common type of system design interview.
	- You might be asked to design the backend for a chat application, or the backend for a ride-sharing application, or design the backend of an application like Slack (referred to by name).
- ==Infrastructure Design==: Less common than product design interviews, but still relatively common.
	- Designing a message broker or a rate limiter.
	- These are deeper in the stack, so the interviewer will be looking for more emphasis on system-level mastery (e.g. consensus algorithms, durability considerations) than high-level design.
- ==Object-Oriented Design==: Sometimes called "low-level design", less common than product design interviews, but still occur at particularly at companies that use an object-oriented language like Java (Amazon is notable for these interviews).
	- You'll be asked to design a system that supports a certain use case, but the focus is assembling the right class structure.
	- You might be asked to design a parking lot reservation system or a vending machine, but you describe the class structure of a solution rather than the services and backend database design.
	- This guide is NOT AS USEFUL for this type OOD interview! Instead, they recommend [Grokking the Low Level Design Interview](https://www.educative.io/courses/grokking-the-low-level-design-interview-using-ood-principles).
- ==Frontend Design==:
	- Frontend design interviews are focused on the architecture of a complex frontend application.
	- These interviews are most common with specialized frontend engineering roles at larger companies.
	- This guide is NOT as useful for a frontend design interview. Instead, they recommend [Great Frontend](https://www.greatfrontend.com/).

Interview Assessment
- Each company will have a different rubric for system design, but they're often similar.
- System design interviews vary by level:
	- Most entry-level software engineering roles will NOT have a system design interview.
	- Once you've reached mid-level, system design interviews become more common.
	- At the senior level, system design interviews are the norm.
- The difference in leveling is most frequently the depth of the solution and your knowledge.
- While all candidates are expected to complete a full design satisfying the requirements:
	- A mid-level engineer might do this with 80% breadth and 20% depth
	- While a senior engineer might do this with 60% breadth and 40% depth

==The most common reason for a candidate to fail a system design interview is not delivering a working system. This is often due to a lack of structure in their approach. We recommend following the structure outlined in the Delivery section.==

Problem Navigation
- Your interviewer is looking to assess your ability to navigate a complex problem.
- This means that you should be able to break down the problem into smaller, more manageable pieces, prioritize the most important ones, and then navigate through those pieces to a solution.

==The most common ways that candidate fail with this competency are:==
- Insufficiently exploring the problem and gathering requirements.
- Focusing on uninteresting/trivial aspects of the problem vs the most important ones.
- Getting stuck on a particular piece of the problem and not being able to move forward.

==High-Level Design==
- With a problem broken down, your interviewer wants to see how you can solve each of the constituent pieces.
- This is where your knowledge of the ==Core Concepts== comes into play.
- You should be able to describe how you would solve each piece of the problem, and how those pieces fit together into a cohesive whole.
- The most common ways that candidates ==fail== are:
	- Not having a strong enough understanding of the core concepts to solve the problem.
	- Ignoring scaling and performance considerations.
	- Spaghetti Design that's not well-structured and difficult to understand.

==Technical Excellence==
- To be able to design a great system, you need to know about bets practices, current technologies, and how to apply them.
- This is where your knowledge of ==Key Technologies== is important. You need to understand how to use current technologies, with well-recognized patterns, to solve the problems.
- The most common ways that candidates ==fail== are:
	- Not knowing about available technologies.
	- Not knowing how to apply those technologies to the problem at hand.
	- Not recognizing common patterns and best practices.

==Communication and Collaboration==
- Interviewers want to see how you work and solve the problem, but also your ability to communicate complex concepts, respond to feedback and questions, and in some cases work together with the interviewer to solve the problem.
- The most common ways that candidates ==fail== are:
	- Not being able to communicate clearly.
	- Being defensive or argumentative when receiving feedback.
	- Getting lost in the weeds and not being able to work with the interviewer to solve the problem.

-----------------
# How to Prepare

### Build a foundation
1. Understand what a system design interview is. Read this guide and watch our videos of [mock system design interviews](https://www.youtube.com/watch?v=tgSe27eoBG0).
2. Choose a delivery framework. System design interviews move fast, and it's important that you have a clear roadmap to help you think linearly **and** avoid scope creep. We strongly recommend our [Delivery Framework](https://www.hellointerview.com/learn/system-design/in-a-hurry/delivery), which is the framework you'll follow to design your system come interview day.
3. Start with the Basics: If you're new to SD, you'll start by learning the basics and mapping out the scope of knowledge required.
	- Start by reading the Core Concepts, Key Technologies, and Common Patterns used in system design interviews. these write-ups are high-level but help you build the mental model that you can build upon.

### Practice, Practice, Practice
- ==YOU WILL RETAIN 10x MORE INFORMATION BY ACTUALLY DOING, RATHER THAN CONSUMING PASSIVE CONTENT!==
1. Choose a question
2. Read the requirements
3. Try to answer on your own
4. Read the answer key (once you've tried to answer your question)
5. Put your knowledge to the test: Once you've done a few questions and are feeling comfortable, put your knowledge to the test by scheduling mock interviews with an interviewer from your target company.

--------------------
