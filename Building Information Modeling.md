---
aliases:
  - BIM
---
Both a process and a ==data format for representing buildings and infrastructure as intelligent 3D models where every element carries structured data== (so not just geometry, but metadata about *what the element is, what it's made of, who made it, and how it relates to other elements*).

A BIM model isn't just a mesh, and a wall isn't just a geometry -- it knows it's a wall, knows its material, knows its fire rating, its thermal properties, which floor its on, which rooms it separates, which manufacturer made it, what its cost is, etc.

[[Industry Foundation Classes]] (IFC) is the open standard file format for BIM data, maintained by buildingSMART International. It defines a schema for representing buildings with hundred of typed entity classes.

Often the geometric foundation of building digital twins:
- Take a BIM model, [[Georeference]] it, stream it as [[3D Tiles]] into [[Cesium]], then overlay real-time sensor data: occupancy sensors, energy meters, HVAC state, access control events, etc.
- The BIM geometry gives you spatial context, while the IoT data gives you real-time state.
- CHALLENGE: BIM models are designed for construction workflows, not real-time visualization; they're often enormous (GBs), unoptimized for rendering (thousands of small objects, complex geometries), and poorly georeferenced. 

Major Software:
- Autodesk Revit: Dominant BIM authoring tool in architecture and construction
- ArchiCAD: Strong alternative in Europe
- Bentley OpenBuildings: Strong in infrastructure (bridges, roads, rail)
- Tekla Structures: Specialized for structural steel and concrete etailing
- Navisworks: Autodesk's clash detection and coordination tool.


# Connection to Geospatial
BIM and GIS have historically been separate domains, but they're converging:
- Georeferencing BIMs (placing a BIM model at real world coordinates so it can be visualized alongside GIS data, like in [[Cesium]])
- [[CityGML]] (an [[Open Geospatial Consortium|OGC]] standard for representing cities as semantic 3D models, similar to IFC but at urban scale)
- [[IndoorGML]] ([[Open Geospatial Consortium|OGC]] standard for indoor spatial information, relevant for indoor routing, emergency response, facility management)
- [[GeoJSON]] + Height: The quick-and-dirty approach. Building footprints with a height attribute, extruded to create simple 3D building representations.