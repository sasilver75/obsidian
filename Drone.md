References:
- Video: [Drones | How do they work?](https://www.youtube.com/watch?v=N_XneaFmOmU)

- Fixed Wing: 
- Multi-Rotor: An aircraft that flies not by air traveling over wings, but by propellers forcing the air downwards. 
	- Most common rotor type is a quadcopter (4), but there are also bicopters (2), tricopter (3), hexacopters (6), octocopters (8), etc.
	- 
# FPV Drones


Components
- Parts that Fly
	- The Frame: Holds everything together
	- The Flight System: Everything required to give the drone to ability to fly and stay aloft
		- ==Battery==: Since weight is a major concern, and high-performance drones require massive power delivered in shot bursts, our options are essentially limited to [[Lithium Polymer]] (LiPo) for high discharge rates or heavier [[Lithium Ion]] (Li-Ion) for extended flight times but less aggressive maneuvering.
		- ==Propellers==: "Props" blow air in one direction at a time, producing thrust in the opposite direction. There are different sizes and shapes of props, and which one you use depends on the size of drone the "flight feel" you're looking for.
		- ==Motors==: High-performance motors used for FPVs are typically [[Brushless Motor]], which can spin at speeds exceeding tens of thousands of revolutions per minute.
		- [[Electronic Speed Controller]]s (ESCs): Used to control the power sent to the motors. Gives the motors the ability to run smoothly and reliably at a wide range of different speeds. Each motor needs its own individual ESC. Sometimes you get 4 ESCs on a single device, called 4-in-1 ESCs.
		- [[Receiver]]: Uses its antenna to receive the pilots commands, and sends them to the Flight Controller. Needs to match the *protocol* of the controller; if you're using an [[ExpressLRS]] (ELRS) controller, you'll need an ELRS receiver on your drone to communicate with it.
		- [[Flight Controller]] (FC): The central unifying component of a drone. Think of it as the "brain", which is able to communicate with every other device attached to the drone. Processes commands from pilots, and translates everything into separate instructions to each motor. 
			- Also features a sensor called a [[Gyroscope]]; helps the drone maintain orientation and corrects for turbulence, wind, drifting, and other external or environmental factors. 
			- Some FCs are also able to process video signals, letting them overlay visual performance data atop the live video that's streamed to the pilot's goggles, such as battery voltage, signal strength, etc.
	- The FPV System: There are two types of FPV systems: ==digital== and ==analog== FPV systems. 
		- Focusing on the similarities, most basic FPV systems boil down to these components:
			- FPV Camera: Sits at the front of the drone, capturing real-time video and sending it to the video transmitter. 
				- Some cameras can stream video to your goggles *and* record higher definition 4k videos simultaneously.
				- Despite this, many pilots still mount a separate Action Camera (e.g. GoPro, Insta360) on top.
			- Video Transmitter: Converts the video signals from the camera into wireless signals. Has an antenna that transmits these signals to the pilot's FPV Goggles.
- Ground Equipment
	- The Remote: Used by the pilot to give commands to the drone; typically has two control sticks, a and a few configurable switches or buttons.
	- The FPV Goggles: Probably the most important part of a great FPV Experience. They have an antenna that receives the wireless video signals from the drone, and a video screen that displays the video feed that's streamed from the FPV camera.
	- Typically need a Charger that's capable of recharging your LiPo batteries.
	- Typically want some basic tools for building or repairing your drone:
		- Quality Hex Drivers (1.5mm, 2mm, 2.5mm)
		- A good soldering iron, along with some solder and flux, which allows you to connect certain components that don't use plug-in connectors, such as the motors.
		- A propeller tool to help you install/replace/remove propellers. Typically this is a M8 nut driver for most standard-sized FPV drones.



