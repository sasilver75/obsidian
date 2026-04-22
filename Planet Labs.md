---
aliases:
  - Planet
  - SkySat
  - Tanager
  - Dove
  - SuperDove
  - Pelican
---
References:
- Video: [April 2024 video of Will Marshall (Planet CEO) @ Stanford, re: AI in EO](https://www.youtube.com/watch?v=cDYaFGE3_ac) (2024)
- Video: [PSW Science: Revolutionizing Earth Observation with Small Sats and AI](https://youtu.be/ql5Cfwwb1pc) (Late 2025)

Now just called Planet, an SF based Earth observation company founded in 2010 by ex-[[National Aeronautics and Space Administration|NASA]] scientists.
- SkySat is Planet's 50cm tasking capability, absorbed from Terra Bella/Google

Philosophy: Flip the traditional satellite model: Instead of a few expensive, large satellites, build hundreds of cheap small ones.

The Constellation:
- ==Doves/[[PlanetScope]]==: The main constellation, with ~200+ 3U cubesats, each about the size of a shoebox. They fly in a "flock" in [[Sun-Synchronous Orbit|Sun-Synchronous]] orbit @ ~475km altitude. Because there are so many, they collectively mage the entire Earth's landmass each day at 3-4m resolution.
- ==SkySat==: A fleet of ~21 larger satellites capable of 0.5m resolution, competitive with [[Maxar]], also capable of short video clips from orbit. Taskable.
	- "I think that SkySat stuff is generally imaged more [[Nadir|Off-Nadir]], and [[Maxar]] satellites... are typically not imaging as far Off-Nadir, perhaps because they're higher." ([source](https://christopherren.substack.com/p/krishna-talks-high-resolution-imagery))
- ==SuperDove==: The current generation Dove, upgraded to 8 spectral bands (added coastal blue, yellow, red edge, and [[Near Infrared|NIR]] bands on top of the original RGB+[[Near Infrared|NIR]]). Better for vegetation and agricultural analytics than the original 4-band Doves.
- ==Pelican==: Their newest generation, announced in 2023, targeting 0.3m resolution, pushing further into Very High Resolution (VHR) territory. First handful launched in late 2025.


Value Add:
- The daily revisit at global scale is pretty unique.
- No other operator can show you a cloud-free-ish image


![[Pasted image 20260415210327.png]]
Above: Planetscope imagery


![[Pasted image 20260415210300.png]]
Above: Pelican image of aircraft in Qatar (Doha), October 2025


![[Pasted image 20260418155442.png]]



![[Pasted image 20260421220403.png]]
Above: From Will Marshall's Video, April 2024

![[Pasted image 20260421220555.png]]
Above: From Will Marshall's Video,, April 2024
- They primarily image the land masses; some ocean areas (e.g. south china sea), but primarily land.

![[Pasted image 20260421220641.png]]
- Human activity (Defense, intelligence use cases, providing/shedding light on events)
- Tracking floods, draughts, renewable energy

![[Pasted image 20260421221205.png]]
Above: "We have.... 4,000,000 images coming down from our satellite every day, which is rather too many to look at in person."

![[Pasted image 20260421221351.png]]
Building by building damage assessment across all of Ukraine, to help understand what sort of cost it will take to rebuild Ukraine.... for the UN. Took a while to get to work!

Subsequently, by the time that we got to the fire in La'Haina, they were able to do it in 24 hours, so the American Red Cross was using it for their decision making about where to bring Medical supplies, etc.

![[Pasted image 20260421221523.png]]
Above: Another analysis of Ukraine, of how every farmer's field was doing across the country.
- What was the crop type?
- How was the yield?
- Overlaid whether it was under Russian control or Ukranian control, to help with food security issues.
	- A lot of other countries relied on Ukranian food supply; it was critical to know where it was and who was effecting it.

![[Pasted image 20260421221659.png]]
Example of looking for the Chinese spy balloon; after shoot-down, they knew where it was, so the question was "where had it come from?"
- We sketched the balloon using hand-sketching... what it might look like. 
- Took two equivalent Earth land mass areas (because we had to trace back through time stack, using wind patterns) to find where it came from.


![[Pasted image 20260421222425.png]]
We're going to be publishing a 3m resolution map of the world's forest carbon, looks at the tree species type, canopy structure, canopy width, and then a load of ground truth using LiDAR in specific forestry regions to get an estimate of the amount of carbon.
- So they're publishing this down to basically the tree level on earth.
- Lots of effort to do carbon financing, carbon markets, and to underpin them you need good data!
- Most of the challenges of carbon markets have been: Accurate but unscalable (tape measures around tree trunks), or global but not ver accurate, which has led to a lot of greenwashing.


![[Pasted image 20260421222607.png]]
Free for students up to a certain amount.
Special partnership with Stanford... certain volumetric restrictions, so if you start downloding the entire databsae, you might come up with limits, etc... but the goal is to allow people to understand the planet, given the mission of Planet.

![[Pasted image 20260421222650.png]]
Growing publication count using Planet data.



From PSW Science Appearance, stuff below.
![[Pasted image 20260421233804.png]]
- The cost to launch has gone down 4x from SpaceX!
- Miniaturization of electronics; satellites can be smaller for equivalent capability. This is the dominant factor
- These are synergistic.


![[Pasted image 20260421234241.png]]

![[Pasted image 20260421234258.png]]


![[Pasted image 20260421234311.png]]
Complete fleet today


![[Pasted image 20260421234321.png]]
~ 150 of these in orbit today, largest EO constellation. 
More or less mirrors tha [[Landsat]] and [[Sentinel|Sentinel-2]] bands, in the optical and [[Near Infrared|NIR]] spectrum
Images 200M sq km a day (Earth's land mass is about 150M sq km, so we do all of that as well as some various parts of the ocean (hot spots in mediterranean, caribbean, south china sea))
We take 4M 47 MegaPixel images per day, a gargantuan amount of data (45 Terabytes a day, in its most compressed form).

![[Pasted image 20260421235218.png]]


![[Pasted image 20260421235226.png]]
Next-generation high-resolution satellite
- They call them 30x30x30
	- 30cm resolution, 30 visits per day, 30 minutes from task request to get the image back (super important for disaster response, security applications). 

![[Pasted image 20260421235315.png]]
Our first hyperspectral instrument in orbit
- Excited by this one; if you really want to understand things, taking spectra is one of the best ways of doign so.
- This is a 400 spectral band instrument, developed by NASA-JPL, and... the first one cost about $50M for the instrument, and we've taken that instrument, figured out how to produce ti more effectively, and built this, such that that cost of the WHOLE  spacecraft, including instrument, spacecraft, and launch is $6-7 million dollars.

![[Pasted image 20260421235439.png]]
Public-Private partnership
- NASA JPL provided that instrument technology, and Planet provided the spacecraft development
- Carbon Mapper is a nonprofit that developed a portal for looking at methane data... but they also provided the funding for this; it's a philanthropically funded mission, and fun way in which PPP can happen.
- The tanager ccollaboration worked well!

![[Pasted image 20260421235609.png]]

![[Pasted image 20260421235621.png]]
Showing just a few of the spectral bands

![[Pasted image 20260421235630.png]]
This was Tanagers mission, to look at methane gas, and where to find them.
- Example: Releasing locations where we see gas emissions
	- They've released 3,000 or so locations.
	- As soon as they put this stuff online, someone realized they had a huge gas leak (7,000kg an hour of methane, which is roughly about the same as 10,000 cars driving full speed, or 100,000 or 200,000 cars if you take into account the duty cycle of car use).
	- If you could do this about 100x per month, you could take out about the US number of cars.
- They got down to 66 kilograms of methane per hour, even more sensitive than they thought.
	- That's like having 4 houses turn on their gas oven and letting it emit.

![[Pasted image 20260421235821.png]]

![[Pasted image 20260421235859.png]]
California is paying for data over california.


![[Pasted image 20260422000109.png]]
Stopping Marijuana grows in CA for instance.
- It's legal in Humboldt, but you need a license
- An interesting way to do automated enforcement of Permitting across states, for instance!

An interesting use case in insurance:
![[Pasted image 20260422000145.png]]
They look at soil moisture data in the top 5-10cm of soil, and basically if it's less than X amount of water for Y days, they send out the adjustment, instead of having to send out a person.


![[Pasted image 20260422000224.png]]
Deforestation timelapse

![[Pasted image 20260422000231.png]]
AI system to look over the Amazon for NEW ROADS, which are usually the sign for deforestation.
Looks over once a week, and provides alerts to the Brazillian federal police, who intervene in that situation.
- ==They seized $2B of assets in Brazil, and reduced deforestation YoY by 55%!==
	- ==Real time information with AI analysis, together with cooperative governments, can go a long way, very fast with this tech!==

Forest Carbon Planetary Variable:
estimating the amount of Carbon in trees:
![[Pasted image 20260422000347.png]]


![[Pasted image 20260422000440.png]]
Lots of work in security too! Bringing transparency and accountability to global situations!

![[Pasted image 20260422000503.png]]
Russian forces erecting bridges across from Belarus into Ukraine
- from day -1, we were calling ttheir bluff on everything that happened.

![[Pasted image 20260422000522.png]]
With MSFT and the UN, estimating the damage across all of Ukraine using automated damage assessment tools.
Subsequently, we use... on the regular basis in Lahaina; in Ukraine it took 2 months, in Lahaina, it took 2 hours.
![[Pasted image 20260422000549.png]]

![[Pasted image 20260422000625.png]]
Back to Ukraine: Field by field, did an assessment with friends @ NASA Harvest on the Crop Yields, Crop types in each field; what fraction was under Russian control and where it was taken:


![[Pasted image 20260422000729.png]]
War Crime analysis: Mass grave found today, what happened last week?


![[Pasted image 20260422000748.png]]
Bringing accountability to Warfare is a whole new thing!


![[Pasted image 20260422000846.png]]
Digital public good, done with the [[Allen Institute]]. 
Able to identify the different types of corals underwater, most are under 15m, so we can still tell what's up.
Allen Coral Atlas dot org


![[Pasted image 20260422000946.png]]
Tracking deforestation, for instance.

![[Pasted image 20260422001011.png]]
Mapping the global renewable facilities (solar, wind facilities) and being able to map them.
- Some countries had totally uncalibrated estimates of how much solar/wind they had.


![[Pasted image 20260422001040.png]]
You can zoom in to the exact building, etc... 

![[Pasted image 20260422001138.png]]
What if we could detect every object on the earth every day, and turn it into a  database that you could search for all objects over time, and you could have a ==queryable earth==!

Imagine asking semantic questions of this, or even predictive questions of it!
![[Pasted image 20260422001626.png]]













