A terrestrial antenna facility that communicates with [[Satellite]]s, receiving data they've collected and sending commands up to them.
- Ground stations are a satellite's only phone call home; when it's overhead, download everything fast and receive instructions (tasking, attitude) for next orbit.

Satellites in [[Low Earth Orbit|LEO]] move fast (~7.5 km/s) and have limited onboard storage; They can only transmit data when they have line-of-sight to a receiving antenna on the ground.

A single ground station might only get 10-minute contact windows, several times per day as the satellite passes overhead.

During a pass:
1. Satellite comes over horizon
2. Ground station antenna tracks it across the sky
3. Satellite downloads collected imagery/data at high bandwidth
4. Ground station uplinks commands for the next orbit (tasking, attitude adjustments)
5. Satellite disappears below horizon, and contact is lost


Polar ground stations (Alaska, Norway, Antarctica) are valuable because [[Polar Orbit]]ing satellites pass over the poles every orbit. Svalbard (Norway) is one of the most important ground station locations on Earth for this reason.

Rather than building their own, many satellite operators lease access to networks:
- [[Amazon Ground Station]]: Amazon's pay-per-minute ground station as a service
- [[Azure Orbital]]: Microsoft's equivalent
- [[KSAT]]: Norwegian commercial network, dominant in polar coverage
- Leaf Space, Atlas Space Operations, etc.: Smaller commercial networks.

New models (AWS, Azure) pipe satellite downlink directly into [[Blob Storage|Object Storage]], so data goes into an S3 bucket in near-real-time without a human touching it, which enables cloud-native [[Remote Sensing|Earth Observation]] pipelines.