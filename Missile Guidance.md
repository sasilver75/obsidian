---
aliases:
  - Active Guidance
  - Semi-Active Guidance
  - Passive Guidance
  - Homing Guidance
  - Remote Control Guidance
  - Command Guidance
  - Beam Riding
  - Semi-Active Radar Homing
  - SARH
  - Active Radar Homing
  - ARH
---
References:
- [Video: Mscope: How Missiles Work (Engines, Guidance, Warheads, and History)](https://www.youtube.com/watch?v=_ahR6LqnyYg)

https://en.wikipedia.org/wiki/Missile_guidance

Guidance systems can be broadly divided into two categories:
- ==Go-onto-target== (GOT) systems, which can be used against moving targets
- ==Go-onto-location-in-space== (GOLIS) systems, which attack fixed geographical positions

Missiles and guided bombs generally use similar types of guidance systems, the difference being that missiles are powered by onboard engines, whereas guided bombs rely on the speed of the launch aircraft and gravity for propulsion.


# Go-onto-target (GOT) systems

## [Homing Guidance](https://en.wikipedia.org/wiki/Homing_guidance#Guidance_laws)
- The missile tracks the target using its own sensors, and uses that information to generate its own control commands.
- Typical sensors include infrared, radar, and light.
- Do not typically need to communicate with a ground station or launch platform, which is useful for fire-and-forget missiles.
- Homing guidance can be divided into three categories:
	- ==Active Radar Homing (ARH)==: The missile illuminates the target using its own source of radiation, such as an on-board [[Radar]]
	- ==Semi-Active Radar Homing (SARH)==: Relies on an *external* source of radiation, separate from the missile (could be on aircraft, or ground)
	- ==Passive guidance==: Tracks only the targets own emissions or contrast against the background.
- Most modern homing missiles use variants of [proportional navigation](https://en.wikipedia.org/wiki/Proportional_navigation) to steer during terminal phase of the flight.
- Some examples of missiles using homing guidance include:
	- [[AIM-120 AMRAAM]] (active homing)
	- [[AIM-7 Sparrow]] (semi-active homing)
	- [[FIM-92 Stinger]] (passive guidance)


### Remote Control Guidance
- Usually need the use of radars and a radio, or otherwise a wired link between the control point and the missile.
- The trajectory is controlled with the information transmitted via radio, beam, or wire (e.g. a "wire-guided missile").
- Sometimes missiles will use Command Guidance (below, subtype) during the boost and middle phases of flight, the switch to homing guidance in the terminal phase. ==HUH==
#### [Command Guidance](https://en.wikipedia.org/wiki/Command_guidance)
- A system in which the guidance commands originate *outside the missile*, requiring two links between missile and transmitter.
	- ==Information link== allows the controller to determine the position of the missile
	- ==Command link== allows commands to be transmitted from the controller to the missile
- A disadvantage of command guidance is that it requires the target to be illuminated by an external energy source, from the launcher or elsewhere. This can alert the target, which could then conduct evasive maneuvers.
- Examples:
	- [[MIM-104 Patriot]]
	- [[S-300|S-300P]]

#### [Beam Riding](https://en.wikipedia.org/wiki/Beam_riding)
- An electromagnetic beam of some sort (either [[Radar]] or Laser), which is pointed at the target. 
- Sensors on the rear of the missile receive the beam and the control systems of the missile use this information to calculate steering commands, attempting to keep the missile in the beam.
- Beam riding systems are often [[Semi-Automatic Command to Line of Sight|SACLOS]], but don't have to be. In other systems, the beam is a part of an automated radar tracking system.
- Advantage is that multiple missile may be launched at once, using the same beam.
- Suffers from the inherent weakness of inaccuracy when increasing range as the beam spreads out; laser beam riders are more accurate in this regard, but can be degraded by bad weather.
- 