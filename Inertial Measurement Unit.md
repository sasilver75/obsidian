---
aliases:
  - IMU
---
A sensor (unit) that measures motion, specifically acceleration and rotation, without reference to any external signal.
- "An IMU is dead reckoning in hardware; it tracks every movement from a known starting point, accumulating error the whole way."

By integrating acceleration over time, you get velocity. Integrate again, and you get position. By integrating rotation rate you get orientation. Starting from a known position/attitude, an IMU tracks where you've gone purely through math, with no GPS needed.
- The catch is *drift:* Every integration accumulates error; small sensor noise compounds into position drift over time. This is why IMUs are almost always fused with GPS or other corrections.

What's Inside:
- Accelerometer: Measures linear acceleration along X/Y/Z axes
- Gyroscope: Measures angular rotation rate around X/Y/Z axes
- (Higher-end IMUs) Magnetometer (compass): Sometimes called a 9-DOF IMU


Relevance to Geospatial:
- For Airborne [[Light Detection and Ranging|LiDAR]], the IMU is what makes point clouds work; the LiDAR measures distance to the ground, but you need to know that the exact position AND orientation of the sensor at every laser pulse. IMU + GPS together give you that. Without a high-grade IMU, your point cloud is a smeared mess.
- Mobile mapping: Survey vehicles, backpack scanners, drone LiDAR all depend on IMU
- [[Georeference|Direct Georeferencing]]: Using IMU+GPS to gereference imagery without ground control points.

