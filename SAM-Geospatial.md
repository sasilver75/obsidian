---
aliases:
  - samgeo
---
A Python package that wraps Meta's [[Segment Anything Model]] for geospatial use cases, making zero-shot image segmentation accessible to GIS/EO practitioners without deep learning experience.

Core idea:
- SAM was trained on 11M images with 1B+ masks and can segments any object in any image with no additional training.
- Samgeo adapts this for remote sensing imagery: handling geospatial formats, CRS, projections, and export to vector/raster that SAM doesn't care about itself.

Three segmentation modes:
1. Automatic: SAM segments everything it finds in the image, no prompts needed. Returns all detected regions.
2. Interactive (GUI): Point-and-click in a Jupyter widget via 
3. Text prompt ([[Grounding DINO]]): Natural language queries like "find all buildings" or "Segment water bodie." Grounding DINO handles open-vocabulary object detection, SAM does the precise segmentation. Authors node this mode has limitations and may need tuning.

Adds over raw SAM:
- Handles [[GeoTIFF]] input (reads [[Coordinate Reference System|CRS]], [[Geotransform]], bands)
- Downloads imagery directly, integrating with type service
- Exports results as [[Georeference]]d vectors ([[GeoJSON]], [[Shapefile]]) or raster masks
- Visualizes results interactively in Jupyter/[[leafmap]]








