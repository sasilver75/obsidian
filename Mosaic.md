---
aliases:
  - Photomosaic
  - Mosaicked
  - Mosaicking
  - Composite
  - Compositing
  - Composited
---
Note: Mosaicking and Compositing refer to different things, but I'm putting them together in this note (for now) because they're... kind of similar, in the sense that we're combining multiple images to form one image.
- ==While distinct, these terms are sometimes used interchangably, though in modern EO tools like [[Google Earth Engine|GEE]], they are recognized as distinct methods==.

Mosaicking and Compositing are both techniques used to combine multiple images into a single image, particularly in remote sensing and GIS, but they serve different purposes based on whether they combine images over space or over time.

# Mosaicking
- Spatially assembling images taken at DIFFERENT locations (often at roughly the same time) to create a single continuous image covering a larger area, like stitching together satellite photos to cover a country. Involves registration (alignment) and blending (reducing seams).
- Goal: Create a larger image

# Compositing
- Combines overlapping images of the SAME location using an aggregation function (e.g. median or max) to create a single, high-quality, often cloud-coverage-free image.
- Goal: Create a cleaner image


