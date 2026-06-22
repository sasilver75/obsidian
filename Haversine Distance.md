
The shortest distance between two points on the surface of a sphere, usually used to estimate distance between two coordinates on Earth.

Latitude/Longitude are angular coordinates, so you cannot treat them like flat `x, y` coordinates unless the distances are very small. Haversine distance computes the great-circle distance, meaning the distance along the sphere's surface.

```
a = sin²(Δφ / 2) + cos(φ1) cos(φ2) sin²(Δλ / 2)

c = 2 atan2(√a, √(1 - a))

d = R c
```
where
```
φ1, φ2 = latitudes in radians
λ1, λ2 = longitudes in radians
Δφ = φ2 - φ1
Δλ = λ2 - λ1
R = radius of the sphere (For earth, 6,371 km)
d = surface distance
```

So for NY to London:
```
New York: 40.7128° N, 74.0060° W
London:   51.5074° N, 0.1278° W

Haversine distance ≈ 5,570 km
```

Important precision note: haversine assumes Earth is a perfect sphere. That is a useful simplification, but Earth is closer to an oblate ellipsoid. For casual mapping, nearby-place lookup, or “distance as the crow flies,” haversine is usually fine. For surveying, aviation-grade routing, legal boundaries, or centimeter/meter-level accuracy, use an ellipsoidal geodesic algorithm such as Vincenty’s formulae or Karney’s algorithm.


