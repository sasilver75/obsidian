---
aliases:
  - Two-Line Element Set
  - TLE
  - Ephemerides
  - SPG4
---

A [[Satellite]] ephemeris is a set of data providing the precise position and velocity of a satellite at a specific time. (Plural is Ephemerides)

==An ephemeris is a table or dataset giving the prediction positions of a celestial object over time.==

In satellites, it gives the ==state vector (position and velocity) as a function of time==.
- Position: (X,Y,Z) in some reference frame
- Velocity: (Vx, Vy, Vz)

From this, you can compute:
- Where the satellite is right now
- Where it will be at any future time
- [[Ground Track]] and [[Swath]] coverage
- When it will pass over some target [[Area of Interest]] (AOI)
- Sun angle at [[Acquisition]] time


The most common ephemeris format is a ==TLE (Two-Line Element Set)==, a compact two-line ASCII format:
```
  ISS (ZARYA)                                                                    
  1 25544U 98067A   24001.50000000  .00001234  00000-0  12345-4 0  9990           
  2 25544  51.6400 123.4567 0001234  12.3456 347.6543 15.49815933123456
```
Encodes: [[Inclination]], RAAN, eccentricity, argument of perigee, mean anomaly, mean motion, giving you everything you need to propagate the orbit. ==SGP4== (Simplified General Perturbations 4) is the algorithm (propagator) used to compute a satellite future position/velocity from those elements.


# Precise vs Broadcast Ephemeris
- For [[InSAR]], you need very ==precise== ephemerides, ==broadcast== ones aren't accurate enough to isolate ground deformation from orbital error.
![[Pasted image 20260419185727.png]]
