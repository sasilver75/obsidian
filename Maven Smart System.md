---
aliases:
  - MSS
  - Maven
  - Algorithmic Warfare Cross Functional Team
---
Resources
- Video: [Multi-Domain AI: The Future of Command and Control | CDAO at AIPCon 9](https://youtu.be/yrtDgoqWmgM?si=jjQFE3MGTJptG2qr)
- Video: [Palantir's Al Targeting System Running the Iran War](https://youtu.be/CHLFl26p7Po?si=CxEoTXIwvFCvOOue)
- Video: [Sandboxx: MAVEN AI is running combat ops in Iran](https://youtu.be/7rR-PlHut2w?si=zbPpVP_f7rgsVqd9) (Very Good)

An AI-powered defense platform (primarily) developed by [[Palantir]] that integrates, analyzes, and acts on massive volumes of sensor and intelligence data for the U.S. military, serving as a core "command and control" ([[Command and Control|C2]]) system.

Leverages computer vision and AI to identify targets and recommend actions, compressing the [[OODA Loop]] for military operators.

Capable of ingesting real-time [[Intelligence, Surveillance, and Reconnaissance|ISR]] feeds and applying computer vision to detect/classify and geotag people, equipment, etc.
It can then use this information to nominate thousands of targets for strikes per hour for users who can then turn to Maven's AI tasking asset recommender to identify the right weapon to engage each target, based on a variety of factors, like the most suitable ordinance for the job, platform flying time, weapon loading details, and the whereabouts of friendly personnel and partner forces.
- Once a target, platform, munition have been identified/determined, Maven can communicate directly with troops in the field or even directly with platforms/weapon systems themselves, such as (in 2020) fire orders to an [[M142 High Mobility Artillery Rocket System|M142 HIMARS]] system. In 2023, Maven directly interfaced with army mission command systems like the Advanced Field Artillery Tactical Data System to generate fire missions in Qatar in real combat operations as part of [[Operation Spartan Shield]] and [[Operation Inherent Resolve]]. 
- In June 2025, Maven gained the ability to interface directly with Army's aviation mission planning system, automating the transition from conventional aviation mission planning systems into Maven's [[Common Operating Picture]], creating a one-stop shop for effective flight mission planning with the most up-to-date intelligence available. 
- By January 2026, Maven was already deployed across all major US combatant commands, as well as many allied commands. 
- People often refer to opening and closing [[Kill Chain]]s, which have six distinct steps (Find, Fix, Track, Target, Engage, Assess ([[Find, Fix, Track, Target, Engage, Assess|F2T2EA]])); in real world combat, there's a likelihood that many if not all of these steps will need to be completed by different assets, with potential delays/gaps between them depending on asset positions/proximity. Maven doesn't really employ the traditional Kill Chain Model; it's more of a backbone for a modern approach called [[Kill Web]]s.
- Maven uses a [[Modular Open Systems Approach]], built to work in-concert with other applications, so that it can evolve in the future: In 2025, it was integrated with a DataMinr platform to feed real-time news alerts to support information operations in Africa, to visualize social media sentiment and provide critical insights.



> "I saw stats where normally we would have 2,000 intelligence officers actually trying to do targeting and look at stuff. Now that's 20, and they're doing it in rapid succession as well." - Palantir's Chad Wahlquist






______________


![[Pasted image 20260704200646.png]]
![[Pasted image 20260704200809.png]]



![[Pasted image 20260702135250.png]]
"It's not just one dat feed, it's multiple. Instead of having 8-9 systems to look at, you fuse it into a single visualization tool that lets you select/deselect different types of data, look at different approaches to data, and most importantly, *action*. Once you have a target you want to work into a targeting workflow..."

![[Pasted image 20260702135425.png]]
"Left click, right click, left click, nominate to the board." This detection is then moved into a workflow.



![[Pasted image 20260702133346.png]]
Each column produces a different type of decision-making process. 

In this Kanban board, it seems that they have the following columns:
- Deliberate
	- I see one with a "Time on target is **expired**" label
	- It seems that some of them have a little "warning" icon in the bottom-right, not necessarily the ones with expired.
	- I see one as being labeled as "Plane," one as being labeled as "AM Site 0018", and another labeled as "Computer Vision Detection."
- Dynamic
	- I see labels as C2 Node, TEL, C2 Headquarters Component, Computer Vision detection, etc.
	- I see some warning icons
	- Not sure what this purple one indicates
- Pending Pairing (Assuming this means pairing of targets with a specific friendly platform)
	- I see a Russian OOB Detected, TEL, reported Sighting, ...
- Paired
	- I see one with a "Time on target is **expired**" label
- In Execution
	- I see one with a "Time on target is **expired**" label
- Pending BDA
- Complete


![[Pasted image 20260702135643.png]]
The Pending Pairing cell was clicked on, which produced this popout.

![[Pasted image 20260702135725.png]]
In Maverick.. 

![[Pasted image 20260702135748.png]]
Once you are actioning, we can move into [[Course of Action|COA]] generation, automatically identifying by a number of factors... what the best assets to prosecute a target looks like. Once we have the different approaches, and we select one, we move into how to action a target. We 

![[Pasted image 20260702135856.png]]![[Pasted image 20260702135934.png]]
![[Pasted image 20260702140056.png]]

Once we select one of the approaches...
![[Pasted image 20260702140116.png]]
We then can move directly into how do we action that target.
![[Pasted image 20260702140148.png]]
((See: SMACK))
We go from identify the target, developing a COA, and now actioning that target, all in one system.

![[Pasted image 20260702140235.png]]

![[Pasted image 20260702140254.png]]

