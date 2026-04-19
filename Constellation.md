---
aliases:
  - Walker Constellation
  - Walker Star
  - Walker Delta
---


A coordinated network of satellites working together as a system to provide continuous coverage of an area (or the whole Earth).

A single satellite in LEO passes overhead for ~10 minutes before disappearing over the horizon. To get continuous coverage, you need enough satellites that as one sets, another rises. Constellations are designed so that geometry is always satisfied.

Higher satellite orbits result in a wider footprint, requiring fewer satellites for global coverage, while lower orbit constellations require many more satellites.
- [[Global Positioning System|GPS]] gets global coverage with ~24 satellites at ~20,200km
- [[Starlink]] needs 4,000+ satellites at 550km to do the same for internet

A ==Walker Constellation== is a specific mathematical pattern for arranging satellites in multiple orbital places to achieve uniform, predictable coverage of the Earth, defines by T/P/F (numer of satellites, number of orbital planes, phase factor; how satellites in adjacent planes are offset, relative to eachother).
- [[Global Positioning System|GPS]] is a Walker 24/6/2 (24 satellites across 6 planes, with a specific offset between planes.)
- *"The mathematical recipe for arranging satellites so coverage is as even and predictable as possible.*
- Two main variants:
	- [[Constellation|Walker Star]]: Orbital planes are [[Polar Orbit|Polar]] or near-polar; distributed round the globe like longitude lines, meeting at poles. Good for polar coverage, used by [[Iridium Satellite Constellation|Iridium]]
	- [[Constellation|Walker Delta]]: Planes are inclined but not polar, distributed symmetrically. Better for mid-latitude coverage, used by [[Global Positioning System|GPS]].

Types of Constellations:
- Navigation: [[Global Positioning System|GPS]] [[Globalnaya Navigazionnaya Sputnikovaya Sistema|GLONASS]], [[Galileo]], [[BeiDou]]
- Earth Observation: [[Planet Labs]] Doves/PlanetScope, [[Maxar]], [[Satellogic]], [[BlackSky]]
- Communications: [[Starlink]], OneWeb, [[Iridium Satellite Constellation|Iridium]]
- [[Synthetic Aperture Radar|SAR]]: [[ICEYE]], [[Capella Space]]

[[Revisit Rate]]:
- The key constellation design metric for EO is how often does at least one satellite pass over a given point? [[Planet Labs|Planet]]'s Dove constellation achieves daily global revisits, while traditional single-satellite missions like [[Landsat]] revisit every 16 days.
