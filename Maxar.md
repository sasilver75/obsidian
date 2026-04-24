---
aliases:
  - Vantor
  - Lanteris
  - WorldView
  - GeoEye
---
ReferenceS:
- Video: [November 2025 Interview with Peter Wilczynski, CPO](https://www.youtube.com/watch?v=Uh9UoFcOsQ4)


Note: In 2023 Maxar rebranded Maxar Intelligence as  [[Maxar|Vantor]] and Maxar Space Systems as [[Maxar|Lanteris]]
- Lanteris quickly acquired by PE Advent International in 2023
- Lanteris was then sold to Intuitive Machines for $800M in November 2025.

A satellite imagery company (Acq. PE Company Advent International 2023) that operates a constellation of *==very high resolution* (VHR) commercial optical satellites==:
- WorldView-1 — 0.5m [[Panchromatic]] only, very fast tasking
- WorldView-2 — 0.46m pan, 1.8m multispectral (8 bands)
- WorldView-3 — 0.31m pan, 1.24m multispectral, 3.7m SWIR — the sharpest commercial
 satellite in operation
- ==WorldView==-4 — 0.31m, but failed in 2019
- ==GeoEye==-1 — 0.41m pan, 1.65m multispectral

![[Pasted image 20260415205415.png|1114]]
Above: Images from next-gen WorldView Legion satellites (2024)

Good for:
- Sub-meter resolution means that you can see individual cars, count parked aircraft, assess building damage after disasters.
- Defense/intelligence work; Maxar is a major [[National Geospatial-Intelligence Agency]] (NGA) contractor.
- Precision mapping, urban planning, infrastructure monitoring.
- The ==Maxar Open Data Program== releases free imagery after major disasters (earthquakes, floods, hurricanes).
Limitations:
- Expensive to task commercially (~$15-30/km^2), low revisit for any single satellite, and clouds remain a problem, since it's an optical satellite.


![[Pasted image 20260418155447.png]]



____________

Video: [Why Maxar Became Vantor & A New Vision for Geospatial AI | Geoawesome Talk](https://www.youtube.com/watch?v=Uh9UoFcOsQ4) (GOOD, end of 2025)


==TensorGlobe== product: A lot of the tech used at Maxar to run their business... and pulling that into a platform that allows other organizations to basically get "Vantor in a Box." 
- From a naming perspective, we see it as an elevation of [[DigitalGlobe]] (acq. 2017 by Maxar) vision, which was about taking the physical world and making it accessible to users in a computer screen, mobile device, tablet.
- This is like that but making it accessible to AI systems. The vision is to provide a digital representation of the physical earth that was accessible to both AI systems and human users.

Q: Core Vision of TensorGlobe?
A: One of interoperability
- Vantor provides a global scaffolding to build on top of. It's about providing a consumer-grade [[Basemap]] for the enterprise, and being able to use the collection from our satellites to ground and improve the accuracy, quality, resolution and the interpretability of data from other satellites!
- We've been doing this with [[Umbra]] and [[Satellogic]] for two years now, as well as the entire geospatial industry to help data from multiple providers get fused together and distributed to end-users.
	- The GEGD program does distribution work for basically all of the commercial data that the US government procures, through a single data portal...
	- With TensorGlobe, we're looking to build on work we've done in the "One World Terrain" program, and processing that data into a unified ,living globe, where each pixel coming in (from video feed from ground, space-base assets, video systems, SAR systems)... each of them is sort of "repainting the globe"
	- The hope fr TensorGlobe is that it becomes a schelling point or gravity well for that data
	- One of the key product visions for TensorGlobe is to... simplify the analaysis experience so that they're really going to one place, and all of the data is already here. Not just isn a raw format, but in a processed foram at to!
		- In raster,
		- In vector
		- Spatially aligned
		- Temporally aligned
		- so that where it comes from, they can apply the same analytic capabilities to a normalized representation of the digital world.


Providing situational awareness regardless of what sensor comes from it.

The fracturing of the information ecosystem is one that's really top of mind, when it comes to AI
- In any sort of... large-scale response to an event, the world has become so fractured from a data perspective (social media feeds, beacon telemetry, devices, ...), but the world is still the same world, it hasn't gotten dramatically bigger -- it's the same 3D earth it was 10,000 years ago.


A drone system is always going to have higher resolution than a space-based system.
With ==TensorGlobe==, we want to shift from a competitive-based world where it's not "Are we doing aerial or satellite?" and shift to an AND world, and design it to feed to AI models, as a core part of that vision.
Models can see the latest data layered on top and linked up. We internally talk about putting every pixel in its place, but there are many pixels that aren't spatial... but most images/pixels people are sharing... it's spatial! We want to combine social media data with space-data, and have a world where you're looking at many frames that are ultimately compositing together into some sort of video, for instance!
- ((Can you imagine a totally new video that's the result of combining data))

This sounds like a very interesting product vision!

_______________________


GeoAwesome: [The Earth Observation industry in 2025 by Maxar Intelligence](https://www.youtube.com/@geoawesome.digital) (end of 2024)
- Speaker: VP of Product @ [[Maxar]] Intelligence ([[Maxar|Vantor]])

Maxar is pushing forwards an industry we helped create
- World's first commercial observation satellite from a US-based company (IKONOS in 1999, 26-27 years ago), with decades of mission expertise in this space.
- Remains the leader in high resolution satellite imagery in geospatial data.
	- Expanding on that to become a leader in geospatial insights.
- [[Maxar Geospatial Platform]] (MGP), launched 4 new satellites with 30cm class data, launching some more in the beginning of 2025.
- Hiring best of class talent... to make the transition to hte next era of earth observation.

=="Our customers are drowning in Pixels."==
- 700% increase in remote sensing satellites in orbit in the last 10 years
- <50% of tasked imagery is used by customers (anecdotal)
- Even when looked at, 30% of time is spent on manual data fusion in some GIS workflows
- 100s of new ML frameworks are developed each year, there's not really a silver bullet in the industry right now.


![[Pasted image 20260423220818.png]]
- Customers can order Maxar or Umbra satellite data... for high resolution optical or SAR imagery
- Rapid access programs for customers who need faster, lower latency, more secure access to that data.
- ==We've also started looking at space, images of objects in [[Low Earth Orbit|LEO]] and beyond.==
- From a content perspective, we process our imagery into some of the most advanced geospatial content products on the planet. The most accurate 15cm and 30cm base maps and vector data that we update frequently, the most accurate representation of the earth in 3D, which are used in visualizations and improving positional accuracy... and enabling  automated Co-registration of multiple spatial datasets.
- In analytics, we offer products that can extract more insights from our best in class content, including object and change detection models, thematic layers like [[Land Use and Land Cover|LULC]], etc.

==MGP Pro== is our subscription product that provides access to the platform.

The gap is in closing the loop and transitioning from a geospatial pixel provider to end-to-end analytics, to provide that *predictive*, more ground truth, and near-real time understanding of what's happening, and what's going to happen as we move towards the future.



_____________

Video: [Vantor's CPO Peter Wilcyznski on mapping the entire world in 3D](https://youtu.be/erCV1MJXU3g) (April 2026)


I have this sense... and by mapping between text and embeddings, and images and embeddings, you create a translational layer from images to text, which acts as  bridge for taking next-gen reasoning models that are fluent in text, and letting them use that intuition that they're trained on, and operate on things in the real world. 

Doing things like: How does the waves of gravity affect positional accuracy, how does the movement of plates affect things over decades, etc... having a virtual environment that mirrors the physical environment.

Using a world model, we might be able to do things like ... imagine the world, but without all of the buildings, for instance. Perhaps we could do all sorts of things that  we could usually do with our imagination.

A long winded way of saying... I think we're building the world, trying to keep it as up to date  as possible... so that you can bring other models to it that can imagine future scenarios, do simulations, etc... and imagine what would happen if we changed our zoning laws, etc... you could have world models hallucinate new structures, if all the buildings were made of bricks, etc... 

Historically our digital globe was produced only by our sensors.
A lot of the work that we've been doing on our ==Forge== system, which takes raw data from our sensors and creates that 3D world... is trying to make it so that you can pour data from other places (an iPhone, an android phone, aerial system)... we think of that as putting every pixel in its place... So we're providing almost a scaffolding, and almost all the other data can fit in nicely and be contiguous across the whole globe.
- ==So you can have some high resolution insets, some mid-resolution insets, etc...==
- So how can we externalize this and make this more of a platform that more data can feed into and give us the bets of both worlds.
- ==You're gonna have much more high resolution data that you could collect from the ground than you could collect from space, but space can continuously collect that data at a regular cadence and satisfy a lot of SLAs==
- A lot of the strategy of this forge piece of the platform... is ==bringing in more sources of raw data into that global, grounded world model==.

Tensor Globe
- ==Forge== (Described above): The fusion element of taking data from many sources and producing a unified globe.
- ==Cortex==:  Designed for [[Constellation]] management! If you want to monitor 150 mines for activity, or 250 electricity plants for activity... we can schedule that constellation across a bunch of individual sensors and satellites.
- ==Nexus==: The API layer for accessing the globe and the data used to produce it.
- (==COMING SOON==): Another component which is focused on LOOKING at the globe and understanding what's happening in the world, We've been working closely with the Google team and a lot fo their foundation models and agent development kit to apply reasoning to that globe. What are the patterns happening across very different parts of the world that are connected:
	- Supply chain
	- Some order of battle workflow
	- ... providing understanding.

The initial apps are the thing that becomes a platform... AWS... didn't set out to build AWS at aAmazon... they set out to build Amazon dot com. And to build that in the late 90s, you had to build storage, compute, networking layer, routing systems, DNS infrastructure... that they built to power Amazon dot com, and then productized that platform.

When I think about TensorGlobe, it's about productizing the core operations of Vantor... what we do every day across our 2,000 people, doing:
- Constellation scheduling
- Production
- Quality assurance
- Distribution
- Exploitation
The whole [[TCPED]] intelligence cycle of Tasking, Collection, Production, Exploitation, Dissemination
- We're productionizing that so other organizations can do it!

Q: I'm sure you've seen of those "situation monitoring" type vibecoding projects... it's cool to see that but cooler to see with your level of sophistication.
A: I've loved looking at those, we've worked with some of the people who've been working on them, and it's awesome to see the creativity of what you can do with these tools. One of the things that's true is... we've standardized for the last 25 years on a zoomable map as the substrate that we think of when we think about digital mapping. The thing about AI is that it CAN look at the whole world... I think a lot of the exciting opportunities is applying AI at the very granular level, and rolling up some of these observations or alerts to a global view, MIB style.


From our perspective... we own a constellation of satellites that works in the public cloud environment! Being able to run AI in a public cloud, as opposed to a classified environment, is just a real change, in terms of the compute buildup (OOMs)... The CAPEX expenditure from the top 5 hyperscalers is probably like 7-80% of total US defense spending!

Do we have a compute cluster big enough to do the whole world at high resolution?... who knows.

For so long we've been focused on CV models, and so much of the exciting work in spatial intelligence is "what happens after you identify all of the vehicles in the parking lot?"

Building a talking computer than can look at the pictures and tell you what happen... is not crazy at all anymore.  and it's very different fro the CV segmentation algorithms that we've typically used in satellite image analysis.

It seems that we can do a lot more than just map the number of cars in a building. It's almost the point where we can map the chemical makeup of cars and stuff, hah.

Developing a language and a grammar for spatial data
- ==If you think of an image stack as pictures of a locatin over time, you can almost think of those as words that form a sentence over time.== And that considered together with adjacent stacks for adjacent locations is sort of like paragraphs.
- A lot of things that we focus on are these types of higher abstractions... Fora lot of machines, these are going to be subspace in embedding spaces that have some sort of embedding content.


A lot of the things that these zoomable maps have nailed is that california replaced San Francisco, etc... and that's encoded by hardcoded systems and semantic knowledge graph things... but if we think about events over time. A lot of spatial intelligence will be about anchoring on those patterns and formalizing themm.

One of hte things thats really critical...
The world "molecule" didn't exist, hte world "chemical" didn't exist.... people invented wordsa s science and technollogy advanced.
 As we think about he next genratino of words, we'll need words for these types of spatial abstractions
 - ==E.g. from a "sandia paper" he said soemthing abouat hwo if you see a parking lot and a looped track anda buillding, it's a school.==

==Syntehtic imagery: We're planning of taking a pictutre of this location in 3 hours... let's gnenerate hte image that we think we're going to get, and then show the image that we're actually giong to get, and compare.==


A lot of what we're seeing from an intelligence/tradecraft perspective is a ton of in-silico generation in this foreward-lookign way ... where we're generating all of the images that we anticipate we'll see over the next week... and then diffing reality against various courses of action... and either corroborating our hypothesis or rejecting our hypothesis...


We launched a product last year called  ==Raptor== which is GPS-resilient positioning technology.
- Now, in GPS, satellites have highly accurate atomic clocks, and they say "This is the time! this is the time!"
- And you get that from multiple clocks and you do the trigonometry to figure out where you are.
- In the modern battlespace people are spoofing those GPS signals all the time, or jamming them, so GPS can and often does go down.
- So... in ==Raptor==
	- ==In a battle situation... Raptor is a product we designed to navigate hte way that humans navigate, which is looking around, seeing hte 3D world, andsaying "Based on my knowledge of London and my obseraivations through my glasses on London, this s where I think I am."==
		- ==We did chipping and shipping, where we cut out pieces of the model at high and low resolution, and give it to hte machine or robot and they can compare what they see on the camera to what they have in the little models they've been given. This gives positional accuracy very cloes to GPS, with teh advantage htat you know where you're looking too!'==
		- 

THERES A HUGE DATA ORCHESTRATION ELEMENT OF UPDATING DATA, multimaster version control systems, taking all the data that's coming up and updating the representation of the world, and making sure that everyone knows what versino for the world they're looking at.... similar to git for code, but also for the world.


Last year a partnership with Niantic on using their augmented reality system against our globe, so that you can do augmented reality globally. A thing about GPS is that it's a global positioning system. While its' possible to do inset mapping.. if you want to go offroad (literally), you need a local positioning system (a terrain positioning system), basically. Taking their TPS and making it global requires them to look at a global representation of the world! that might not be important  in super enterprise or consumer worlds... but if you're a company trying to build AR experiences that DO work globally, that's sort of the genesis of that partnership.


It sounds like you've been focusing on making all this information and that you created... accessible to today's language model and reasoning models... but what about vision models?... will there be a version where an agent system will be going in the 3d representatoins that you've created, and how do you think about those use cases?
- yeah, there's a lot of cases for simulation stuff about self driving miles or trying to train robots 


We do most of our perception layer (segmentation, computer vision)... which is mostly done in 2D space. ==A lot of the work we've been doing over the last year is taking a single image or a single stereo-pair image from our satellite and create an on-the-fly 3D image, and have been doing a littl work on the AI perception side to see: How much better does the perception model get when you give it a 3d image versus when you give it a 3D image.==
- ==This is an interesting area of research!==
- So much of this research elsehwre has been about segmenting and perceiving 2D images, but with the third dimension, you're getting ton more information!
- ==A lot of how I as a human interpret a 2D scene is by constructing a 3D scene under the hood==! all living intelligence is about being in a 3D world.

Vantor has an open data program for a lot of thing like wildfire events and things of public notes....
- we run a nubmer of ML exercises and have make pieces of our archives accessible to people who are workign on particular maps, certain academic groups, etc.








