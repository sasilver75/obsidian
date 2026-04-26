---
aliases:
  - RPC
  - Rigorous Sensor Modeling
  - Rational Polynomial Coefficients
---
> How off-nadir displacement is corrected:
  > The sensor's exact position, orientation, and look angle at the moment of capture are known from the satellite's ephemeris data (orbit position) and attitude data (roll/pitch/yaw). Combined with the DEM, you can trace a ray from each pixel back through the sensor geometry to the ground, figure out where it actually hit the Earth's surface accounting for elevation, and resample the image to where that point should appear on a map. This is called rigorous sensor modeling or using an RPC (Rational Polynomial Coefficient) model — a mathematical approximation of that ray-tracing geometry that comes bundled with satellite imagery.
  
![[Pasted image 20260425200440.png]]
