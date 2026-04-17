---
aliases:
  - IFC
---


[[Industry Foundation Classes]] (IFC) is the open standard file format for BIM data, maintained by buildingSMART International. It defines a schema for representing buildings with hundred of typed entity classes:

```
IfcProject
    └── IfcSite
          └── IfcBuilding
                └── IfcBuildingStorey (floor)
                      ├── IfcWall
                      ├── IfcDoor
                      ├── IfcWindow
                      ├── IfcSlab (floor/ceiling)
                      ├── IfcBeam
                      ├── IfcColumn
                      └── IfcSpace (rooms)
```
Each element has:
- 3D geometry
- Property setes (fire rating, acoustic rating, thermal resistance)
- Relationships (connection to adjacent elements, contained in spaces, hosted by walls)
- Materials
- Classification codes (Uniclass, Omniclass)