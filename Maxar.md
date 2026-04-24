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


From our perspective... we own a constellation of satellites that works in the public cloud environment! Being able to run AI in apublic cloud, as opposed to a classified envonrment, is just a real change, in temrso f the compute buildup (OOMs)... The CAPEX expenditure from the top 5 hyperscales is probably like 7-80% of total US defense spending!
	 














