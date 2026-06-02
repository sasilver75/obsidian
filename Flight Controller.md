Sits between the radio [[Receiver]] and the (e.g.) [[Electronic Speed Controller]]s (ESCs) to control how the model flies.

Uses a number of on-board sensors ([[Inertial Measurement Unit]]/IMU) to "feel" how the vehicle is flying ([[Gyroscope|Gyro]]s, [[Accelerometer]]s). By knowing from the radio receiver how you *want* the vehicle to move, and by feeling how it's moving right now, it can constantly make minute adjustments.

Typically a little [[Printed Circuit Board|PCB]] that runs software ([[Firmware]]) like iNavflight, Betaflight, ArduPilot. Most flight controller software also comes with a program to install on your PC that's used to copy/flash/upload the firmware onto the flight controller and set up everything.

Some flight controllers need a separate [[Power Distribution Board]] (PDB), while others are all-in-one (AIO) boards that have a PDB as part of the design for super simple installs.

Different types of Flight Controllers:
- No flight controller is perfect for every build and model type; some are better than others. The more connections on the flight controller PCB, the more things you can connect, like GPS/Telemetry/OSD/Airspeed sensors.
- Most modern flight controllers are made for either multirotores, fixed wing, or both.

Some flight controllers have pre-set modes:
- Angle: Limits the pitch and roll and returns the model to level when you let go of sticks. Like a self-stabilization mode. For beginners/learning to fly, or for gentle cruising.
- Horizon: Like Angle, until you push the controls towards the edge, and then it overrides the limited pitch/roll. Like Angle mode with extras; if you push the sticks to the edge of their travel on your [[Radio Controller]], it will overcome those limits. For when you want to be a little more aggressive.s
- Rate/Manual: The flight controller performs the basic, but it's up to you. For when you don't want any real help from the FC besides the mixing it's doing.
- GPS RTH: 

For the main processor "brain" on the Flight Controller, the ==STM32 MCU family== is the most common one you'll encounter. There are various common labels:
- F4: Still usable, but budget/legacy feeling. Not good if you have lots of peripherals.
- F7: Strong normal choice for FPV, better headroom and I/O behavior than F4.
- H7: Best high-end/future-proof choice, useful for complex builds, lots of UARTs/peripherals, ArduPilot/PX4/iNav-ish builds, or more than 4 motors. Overkill for many basic 5-inch quads.


Common firmware that you'd consider running on your Flight Controller:
- [[ArduPilot]]: An open-source autopilot software stack used for multirotors, fixed-wing aircraft, rovers, boats, and other vehicles.
- [[PX4]]: An open-source autopilot software stack used for drones and robotic vehicles.
- [[Betaflight]]: Flight-control firmware primarily used for FPV multirotors and racing drones.
- [[Intelligent Navigation System|INAV]]: FPV with navigation: GPS rescue/RTH, fixed-wing, long range, simple missions.