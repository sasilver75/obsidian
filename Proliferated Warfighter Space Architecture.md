---
aliases:
  - PWSA
  - National Defense Space Architecture
  - NDSA
---
References:
- [Blog: Payload Space: The Proliferated Warfighter Space Architecture (PWSA): An Explainer](https://payloadspace.com/ndsa-explainer/)

In 2019, the Defense Department stood up the [[Space Development Agency]] (SDA). Originally an independent defense agency, the SDA was given a mandate to move fast to put newly emerging technologies into the warfighter’s hands. The SDA is now part of the [[United States Space Force|Space Force]], but its mission remains the same. 

The National Defense Space Architecture (NDSA; later renamed the PWSA) grew out of the SDA. In essence, the NDSA is a tactical LEO network designed to communicate missile warnings; position, navigation, and timing data; and other vital information to wherever it’s needed on the ground as quickly and securely as possible.

From the SDA, there were multiple tranches of satellites:
- “Tranche 0 (FY22)—Warfighter immersion: The minimum viable product is demonstrating the feasibility of the proliferated architecture in cost, schedule, and scalability towards necessary performance for beyond line of sight targeting and advanced missile detection and tracking.
- Tranche 1 (FY24)—Initial warfighting capability: Regional persistence for tactical data links, advanced missile detection, and beyond line of sight targeting.
- Tranche 2 (FY26)—Global persistence for all in Tranche 1. This will incorporate lessons learned from operating gen 0 for at least two years.
- Tranche 3 (FY28)—Advanced improvements over Tranche 2. This includes better sensitivity for missile tracking, better targeting capabilities for BLOS, additional PNT capabilities, advances in blue/green lasercom and protected RF comm.
- Tranche 4 (FY30)—“Continual advances to the layers, including additional capabilities identified as current or future threats to the warfighter.”

The NDSA contains two separate constellations: the ==Transport== and ==Tracking== layers. 
- The first one is the Transport Layer, which will form a mesh network in LEO connected by optical inter-satellite links. These links, which transmit data via laser, can transmit data at light speed using a very narrow beam that is much more difficult to intercept than traditional radio transmission. All the information transmitted across the architecture will travel through the transport layer. Then, the layer will route it to where it needs to be on the ground.
	- The ==Navigation== layer is not actually a constellation itself, but rather an added benefit of the mesh network formed by the Transport Layer satellites. By nature of the mesh network spanning the globe, the Transport satellites will be able to transmit precise position, navigation and timing (PNT) data. 
		-   “We’re not trying to replace GPS under any circumstance—it’s a core capability that will remain a core capability. But what we’re doing is for those days when GPS is out of band, that we will have the ability for the warfighter to still know what time it is to the accuracy that they need on a global basis, and to know where they are to the accuracy that they need."
	-  Each Transport Layer satellite carries a payload hosting a computer to dynamically manage interactions between the satellites as well as the individual layers. This package is referred to as the ==Battle Management== layer. This layer, which will be hosted on most or all NDSA satellites, is tasked with on-orbit processing (or “edge computing”), which can reduce the pain points associated with relaying data.
- The Tracking layer is the second constellation, and it does the actual remote sensing and Earth observation from LEO. Tracking Layer birds will be fitted with infrared sensors to spot and track missile threats. These satellites will be connected to the Transport Layer through optical links, and their data can be transmitted across the mesh network and downlinked to the ground.
- Intelligence, surveillance, and reconnaissance (ISR) functions are the domain of the Custody Layer. The SDA is not planning to launch its own Custody Layer constellation.
- The final layer of the architecture leaves a little wiggle room for the development of new space technologies.



