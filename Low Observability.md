---
aliases:
  - VLO
  - Stealth
  - Low Observable
  - LO
---
Several factors can contribute to Low Observability:
1. Visual Acquisition (Easily handled by flying by night)
2. Contrails (Easily handled by flying by night)
3. Engine Smoke (Easily handled by flying by night)
4. [[Electromagnetic Spectrum|EM]] Emissions (Deactivating or using receive-only modes for radios, data links, radar altimeters, etc.)
5. Acoustics (By flying at an altitude where the aircraft can't be easily used by the ground)
6. [[Infrared]] (IR) (This actually requires special engineering of the aircraft: Obscuring the view to the hottest part of the aircraft, the engine; place engine intakes and exhaust in a place where it's blocked from view from below.)
7. [[Radar]] (This actually requires special engineering of the aircraft)


> "Stealth does not make an aircraft invisible. Low-observable design reduces the ranges at which radars can detect, classify, track, and engage the aircraft."

> "Lower-frequency radars may detect the presence of a low-observable aircraft but often have poorer angular resolution than higher-frequency fire-control radars with practical antenna sizes."

> "An early-warning radar detecting an aircraft does not automatically mean a weapon can engage it. An engagement requires sufficiently precise location, velocity, update rate, track continuity, and uncertainty bounds. [[Sensor Fusion]] can narrow the gap between detection and engagement by combining measurements from radars operating at different frequencies and from different locations."


# Defeating Infrared
Obscure the view (from below) to the hottest part of the aircraft: its engine.


![[Pasted image 20260718230809.png]]
Above: On the [[B-2 Spirit]], see the fore engine intake and aft engine exhausts, which are all on the top of the plane.

![[Pasted image 20260718230851.png]]
This gives the exhaust air enough time to mingle with cooler ambient air before it's in view.

![[Pasted image 20260718230920.png]]
This limits an aircraft to flying at high altitude, since ther are airborne infrared sensors as well!


# Defeating Radar Reflection

Builders tried use Radar Absorbing Material (RAM), an example of which is a type of paint consisting of a non-conductive substance with tiny mounts of iron suspended in it. This was used in the [[SR-71 Blackbird]], but didn't work incredibly well, compared to the [[B-2 Spirit|B-2]]. Another approach uses conductive particles like crystalline graphite mixed with urethane to form a foam absorber. This is what the pyramids inside an anechoic chamber are made of. But both of thee are limited to absorbing a small frequency range, based on how they're coated.
- So a coating might work against high-frequency fighter radar, but not work against low-frequency early warning radars.

For things like the F-117, the shape of the aircraft has a big effect.
So how does shape affect observability? See [description here](https://youtu.be/_HZYqSXdj5c?si=e5cHXxr1cBXooDlu&t=195)
The short answer is that you control your signature by directing the [[Specular Reflection]] away from the enemy radar emitter, as done in the [[F-117 Nighthawk]], so that the enemy radar only receives the [[Diffuse Reflection]], which is very weak. Also in the design you see the avoidance of [[Corner Reflector]]s. Jet engine exhausts, tail wings, etc are all corner reflectors, and the F-117 removed and minimized as many of these as possible, even at the cost of a non-ideal aerodynamic design, impacted stability, etc.

[[Radar Cross Section]] (measured in meters squared): An expression of how much energy would be reflected from a magical perfectly-conducting sphere of that size (Really, the reflecting part of the sphere, see picture below). It's about the part of the sphere that reflects back to the emitter; the top, bottom, and back don't count.
- A sphere with 1.13m diameter would have 1m^2 of RCS
- A human would be expected to have 1m^2 of RCS
- Small combat aircraft? 2-3m ^2
- A cargo plane a 100 m^2
- A 1.5m [[Corner Reflector]] would have an RCS of 20,000 m^2
	- This makes you understand why the F-117 folks were willing to give up performance to eliminate corner reflectors!

![[Pasted image 20260721150054.png]]

Note that [[Radar Cross Section]], and the amount of reflection, changes with the orientation of the aircraft to the emitter.
- For an F-117, you wouldn't see any [[Specular Reflection]]s from the front, but as it flies above the radar emitter, it exposes the large flat surface of the stabilizers, which will give a large RCS.

A more accurate depiction of this might be like this RCS plot:
![[Pasted image 20260721150238.png]]
But even for this you want to understand it in a three-dimensional world, understanding how it looks above and below the horizon.

RCS plots for most stealth vehicles aren't available, but there is one for the F117:
![[Pasted image 20260721150401.png]]
See that from the sides, the rear stabilizers provide a wide reflecting surface from the sides.


Radar returns are subject to the inverse square law: The decrease in energy from returns doesn't fall off at a linear rate: The return from a target twice as far away is much less than half as strong.
- 