---
aliases:
  - GNSS
---
An umbrella term for satellite-based [[Position, Navigation, and Timing]] (PNT), including examples such as:
- [[Global Positioning System]] (GPS): American
- [[Globalnaya Navigazionnaya Sputnikovaya Sistema]] (GLONASS): Russian
- [[Galileo]]: European
- [[BeiDou]]: Chinese
- [[Quazi-Zenith Satellite System]] (QZSS): Japan
- [[Navigation with Indian Constellation]] (NavIC): India

While GNSS gives raw measurement, the receiver plus correction/estimation method turns those measurements into a position solution; the drone then [[Sensor Fusion|Fuses]] this with onboard sensors:
1. GNSS satellites transmit a precisely time radio signal plus data about its orbit and clock.
2. Receiver measures distance-like values.
3. Receiver computes a baseline position for itself, using signals from several satellites. This solution is often meter-level, because many errors remain.
4. Satellite clock error, satellite orbit error, ionospheric delay, tropospheric delay, multipath reflections, receiver noise, bad satellite geometry, jamming, spoofing.
5. Different systems provide correction data:
	1. [[Satellite-Based Augmentation System]] (SBAS): Regional satellite-broadcast corrections and integrity, such as wAAS or EGNOS.
	2. [[Ground-Based Augmentation System]] (GBAS): Local ground-based corrections, mostly for aviation/airport aproaches.
	3. [[Real-Time Kinematic Positioning|RTK]] base station / [[NRTK]] network: Nearby reference-station corrections for centimeter positioning
	4. [[Precise Point Positioning|PPP]] correction services: Precise satellite orbit and clock corrections, often delivered over satellite or internet.
6. The specific positioning technique determines how the corrections are used.
	1. SBAS/DGNSS: receiver applies code-based corrections for better meter/sub-meter accuracy.
	2. RTK: Receiver uses carrier-phase measurements plus nearby correction data to get cm-level real-time position.
	3. PPK: Same general precision idea as RTK, but processed after the flight.
	4. PPP: Uses precise global corrections without a nearby base station, usually with some convergence time.
	5. PPP-RTK: Hybrid approach aiming for wide area, fast, high-precision positioning.
7. The drone does not blindly fly from GNSS alone; the [[Flight Controller]] fuses [[Global Navigation Satellite System|GNSS]] with an [[Inertial Measurement Unit|IMU]] ([[Accelerometer]], [[Gyroscope]]), [[Barometer]], [[Magnetometer]], [[Compass]], optical flow, [[Visual Odometry]], [[Light Detection and Ranging|LiDAR]], [[Radar Altimeter]]. This fusion is often done with something like an extended [[Kalman Filter]] or similar estimator.


Note: [[Real-Time Kinematic Positioning|RTK]] and [[Precise Point Positioning|PPP]] do not replace GNSS, they are way of processing GNSS measurements with correction data to get better position solutions.
- RTK is typically better when you need centimeter accuracy right now, and can use a nearby base station or RTK network. Depends on local correction infrastructure and degrades when rover gets far from base.
- PPP is usually better when you need high accuracy over a wide area without setting up a nearby base station. Often needs convergence time before reaching bets accuracy.

See also:
- [[Real-Time Kinematic Positioning]] (RTK): High-precision GNSS technique using correction data from a nearby base station to achieve centimeter-level positioning in real time.
	- The main thing to know for cm-level [[Drone]] positioning.
	- [[Network RTK]] (NRTK) uses a network of reference stations instead of a single local base station.
	- Virtual Reference Station (VRS): A network RTK method that computes a synthetic reference station near the rover to improve correction quality.
- [[Precise Point Positioning]] (PPP): High-precision GNSS technique using precise satellite orbit and clock corrections, usually delivered from a global correct service, to improve positioning *without needing a nearby base station.*
- [[Satellite-Based Augmentation System]] (SBAS): A regional correction system, such as WAAS or EGNOS, that broadcasts GNSS correction and integrity data from satellites. This is the "free regional correction/integrity layer" used in ordinary GNSS receivers.
- [[Ground-Based Augmentation System]] (GBAS): A local airport-area correction system providing high-integrity GNSS corrections for precision aviation approaches.
- PPP-RTK: Combines PPP-style global correction with RTK-style ambiguity resolution to provide fast, high-precision positioning over a wide area. Modern receives/services are moving in this direction.
- Real-Time eXtended (RTX): A commercial correction-service approach associated with Trimble that provides PPP-like precise positioning corrections in real time.

Error sources:
- Ionospheric Delay: GNSS error caused by satellite signals slowing as they pass through the ionosphere.
- Tropospheric delay
- Multipath: GNSS error caused by signals reflecting off surfaces before reaching the receiver.
- [[GNSS Jamming]]: Interference preventing a receiver from using satellite navigation signals.
- [[GNSS Spoofing]]: Transmitting false navigation-like signals to mislead a receiver.
- Satellite orbit errors
- Satellite clock errors


==GNSS Frequency/Signal Band==: A named, satellite-navigation signal band, such as L1, L2, L5, or L6, used by GNSS receivers. The carrier frequency is the radio frequency on which a GNSS signal is transmitted. L1 Bands, L2 Bands, L5 Bands, L6 Bands, etc.
- ==Multi-Band GNSS== uses multiple frequency bands like L1/L2/L5/L6. This is central to modern [[Drone]] navigation specs.
- ==Multi-Constellation GNSS== uses GPS, Galileo, BeiDou, etc. together for more satellites and better availability.


![[Pasted image 20260601144441.png]]
Above: Major civil/open GNSS frequency bands Omits military/restricted signals, etc.
- L1 / E1 / B1C: 1575.42 MHz
	- Used by GPS, Galileo, BeiDou, QZSS, NavIC, and SBAS; main mass-market GNSS band for phones, drones, timing, basic navigation, and broad interoperability.
- BeiDou B1I: 1561.098 MHz
	- Used by BeiDou; legacy/open BeiDou navigation signal near, but not identical to, the L1/E1 common frequency.
- GLONASS G1 / L1: approximately 1598-1609 MHz
	- Used by GLONASS; primary GLONASS navigation band.
- L2: 1227.60 MHz; used by GPS and QZSS
	- Precision/survey/RTK band, especially paired with L1 for ionospheric correction.
- GLONASS G2 / L2: approximately 1243-1252 MHz
	- Used by GLONASS; second GLONASS frequency for dual-frequency precision.
- L5 / E5a / B2a: 1176.45 MHz
	- Used by GPS, Galileo, BeiDou, QZSS, NavIC, and SBAS; modern high-performance civil band for aviation, dual-frequency receivers, RTK, and improved robustness.
- E5b / B2I / B2b: 1207.14 MHz
	- Used by Galileo and BeiDou; additional civil/precision band for multi-frequency positioning and BeiDou PPP-B2b correction services.
- E5: 1191.795 MHz
	- Used by Galileo; wideband signal combining E5a and E5b for very precise ranging.
- GLONASS G3 / L3: 1202.025 MHz
	- Used by modern GLONASS; newer GLONASS CDMA civil signal.
- E6 / L6: 1278.75 MHz
	- Used by Galileo and QZSS; high-accuracy correction/authentication-oriented services, including Galileo HAS and QZSS CLAS/MADOCA-PPP.
- BeiDou B3: 1268.52 MHz
	- Used by BeiDou; BeiDou-specific third frequency for multi-frequency precision and redundancy.
- NavIC S-band: 2492.028 MHz
	- Used by NavIC; regional NavIC signal outside the usual GNSS L-band area.
- SBAS L1/L5: 1575.42 MHz and 1176.45 MHz
	- Used by WAAS, EGNOS, MSAS, GAGAN, and other SBAS systems; correction and integrity messages for aviation and improved civil positioning.


