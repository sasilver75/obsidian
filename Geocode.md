---
aliases:
  - Reverse Geocode
  - Geocoding
  - Reverse Geocoding
---


==Geocoding== is the process of converting a human-readable address or place name into a geographic coordinates (latitude, longitude):
```
"1600 Pennsylvania Ave NW, Washington DC" → (38.8977, -77.0365)
```

==Reverse Geocoding== is the opposite: Turning coordinates into an address:
```
(34.052, -118.243) → "Downtown Los Angeles, CA"
```


During ingestion of 311 data in our LA Observatory project, most records already had lat/lon coordinates, but some records have either missing or invalid coordinates; using geocoding (turning an address into a coordinate) here would be the thing to do.

Alternatively, if we were to add a search bar UI to a map, where a user types an address into the search bar, we'd send it to a geocoding service (e.g. [[Nominatim]], which uses [[OpenStreetMap]] data) to get back a lat/lon, then fly the map to that location.


