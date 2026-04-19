---
aliases:
  - GPS
---
A [[Global Navigation Satellite System]] (GNSS).
The US-Operated Satellite [[Constellation]] that provides positioning, navigation, and timing (PNT) anywhere on Earth.
- Ground truth for [[Remote Sensing]]; GPS-tagged field observations validate satellite classifications. Used for [[Georeference|Georeferencing]] of imagery/point clouds.

# Accuracy
- Consumer GPS is accurate to ~3-5m
- WAAS/SBAS (involves differential correction via ground stations): ~1m
- [[Real-Time Kinematic GPS]]: centimeter level, used in surveying, precision ag, AVs
- [[[Precise Point Positioning]] (PPP): Also centimeter level, but slower to converge.

# How it works:
- 24+ satellites in [[Medium Earth Orbit]] (~22,200km), arranged so that 4+ are always visible from any point.
- Each satellite continuously broadcasts its position + precise atomic clock timestamp.
- Your receiver measures the *time of flight* of signals from multiple satellites
- With 3 satellites you can get your 2D position; a fourth satellite resolves altitude and corrects for receiver clock error.
	- This is called *trilateration* (not triangulation; you're measuring distances, not angles)
	- Time of flight from satellites -> distance spheres -> where they intersect is you


