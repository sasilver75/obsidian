
A GPU-powered framework for visual exploratory data analysis of large datsets.
- React friendly
- Interactions with Base Map providers; works nicely with your favorite base map libraries like Google Maps, Mapbox, ArcGIS, MapLibre, and more.


Uses [[WebGL]] directly, meaning it can handle millions of data points at 60fps in a way that regular DOM-based charting libraries like [[D3]] or  Chart.js cannot.

In my project, I'm using [[MapLibre GL JS]] for the base map layer, and deck.gl for the data layers on top of it, attaching them to the same canvas as an overlay via an adapter called `MapboxOverlay`.


Another alternative is Leaflet, which is another popular open-source map library, but it's SVG/DOM-based -- so its' fine for hundreds of features, but not for hundreds of thousands. And unlike deck.gl, it doesn't have things like `H3HexagonLayer`built in.  D3.js is powerful but isn't really designed for geographic projection at map zoom levels, and can't display as much data.