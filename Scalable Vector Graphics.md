---
aliases:
  - SVG
---
SVG is a subset of the [[Document Object Model|DOM]] -- it's an [[Extensible Markup Language|XML]]-based format for describing 2D graphics using geometric primitives.

You write markup like this:
```html
<svg width="400" height="200">
  <circle cx="100" cy="100" r="50" fill="steelblue" />
  <rect x="200" y="50" width="100" height="100" fill="orange" />
  <line x1="0" y1="0" x2="400" y2="200" stroke="white" />
  <path d="M 10 80 Q 95 10 180 80" stroke="red" fill="none" />
</svg>
```

The key word is ==scalable==: XML describes shapes mathematically (a circle at position x, y with radius r), not as a grid of pixels.
- ==Resolution Independent==: Zoom in as far as you want, and ==lines stay sharp==.
- Styleable with CSS
- ==Part of the [[Document Object Model|DOM]]:== Every \<circle\> is a real DOM node; you can attack click handlers, animate with CSS, query with `document.querySelector`.


Tools like [[D3]].js generates SVG based on your data.
- A D3 bar chart is literally a bunch of \<rect\> elements whose height attribute is bound to data values.


## SVG is not a good fit for maps
- Each SVG element is a DOM node
- A city with 50,000 building polygons = 50,000 DOM nodes, and abrowser's layout engine has to track all of them.
- Past 10,000 nodes or so, the performance collapses (laggy, stuttering, etc). 
- Instead, you should use [[WebGL]]-based tools, which bypass the DOM entirely and use the GPU for parallel processing.




