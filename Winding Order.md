The direction in which the vertices of a polygon ring are listed, either clockwise or counterclockwise. It sounds trivial but it's a pervasive source of subtle bugs in geospatial data.

A polygon with vertices A, B, C, D can be defined two ways:
```
Counterclockwise (CCW):  A → B → C → D → A                                Clockwise (CW):          A → D → C → B → A
```
 
 The ***standard*** convention in ***most*** geospatial contexts is the right-hand rule:
  - Exterior rings (the outer boundary of a polygon) → counterclockwise                             
  - Interior rings (holes) → clockwise
If you curl the fingers of your right hand in the direction of vertex traversal, your thumb points toward the "inside" of the polygon. CCW = thumb points up = exterior. CW = thumb points down = hole.

Different standards use different conventions, and they contradict eachother:
- **GeoJSON (RFC 7946)** — exterior rings CCW, holes CW (right-hand rule)       
- **Shapefile** — exterior rings CW, holes CCW (opposite of GeoJSON)
- **WKT/WKB** — technically unspecified in the original spec, implementations vary.       
- **OpenGL/WebGL** — CCW is front-facing by default                                                     
- **3D graphics generally** — winding order determines which side of a face is "outward" facing, affects backface culling.


  ==You rarely think about winding order explicitly when using high-level tools — Shapely, GeoPandas, PostGIS all handle it internally. It surfaces when writing custom code.==