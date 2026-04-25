Uber's ==GPU-powered== data visualization framework, which renders data layers on top of a base map.
- ==While something like [[MapLibre GL JS]] might be used to draw the map itself, [[deck.gl]] is going to be used to draw your data.==
- Interactions with Base Map providers; works nicely with your favorite base map libraries like [[Google Maps]], [[Mapbox]], [[ArcGIS]], [[MapLibre GL JS]] and more.

![[Pasted image 20260424173834.png]]

Uses [[WebGL]] directly, meaning it ==can handle millions of data points at 60fps== in a way that regular DOM-based charting libraries like [[D3]] or  Chart.js cannot.

Everything in deck.gl is a `Layer`, which is an immutable description of how to render a dataset. When data changes, React creates a *new layer instance* and deck.gl diffs it against the old one, uploading only what changed to the GPU.
- `H3HexagonLayer`, for instance, takes H3 cell IDs as strings, and knows the geometry of every H3 cell on Earth.

Another alternative is [[Leaflet]], which is another popular open-source map library, but it's SVG/DOM-based -- so its' fine for hundreds of features, but not for hundreds of thousands. And unlike deck.gl, it doesn't have things like `H3HexagonLayer`built in.  D3.js is powerful but isn't really designed for geographic projection at map zoom levels, and can't display as much data.
- It's older, simpler, and great for basic maps.