---
aliases:
  - Geosynchronous
  - GSO
---

- Platforms orbiting at 35,786km are at an altitude at which their orbital speed matches the planet's rotation, and are in what is called [[Geosynchronous Orbit|Geosynchronous]] ([[Geosynhronous Orbit|GSO]]) orbit.
	- A platform in GSO directly above the equator will have a [[Geostationary Orbit|Geostationary]] orbit, which enables a platform to maintain its position directly over the same place on the Earth's surface. So ==Geostationary is a subclass of Geosynchronous==.
	- While both [[Geosynchronous Orbit]]s and [[Geostationary Orbit]]s are at 35,786km above the Earth, geosynchronous platforms have orbits that can be tilted above or below the equator, while geostationary platforms orbit Earth on *the same plane as the equator*.


> Q: So why would you want a Geosynchronous orbit that isn't Geostationary? It seems like that "floating above one point on the earth" quality is pretty useful.
> A: The main legitimate use cases/reasons are ==fairly narrow==:
> 	- Better high-altitude coverage (A geostationary satellite at 0 degree inclination has a terrible view about ~60 latitude).
> 	- Antenna footprint shaping: Some communication satellites use a slight inclination deliberately so that their coverage footprint slowly shifts north and south throughout the day, spreading wear on transponders.
> 	- Graveyard/transition orbit: Satellites drifting toward or away from a geostationary slot pass through inclined geosynchronous orbits. End-of-life satellites that weren't fully boosted to the graveyard ring can end up in slowly-inclining geosynchronous orbits as residual forces act on them over years.
> 	- Honest answer: ==Truly intentional geosynchronous orbits are rare. Geostationary is almost always what operators want.==