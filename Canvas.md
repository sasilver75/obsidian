
There are three ways that browser can draw graphics:
1. The [[Document Object Model]] (DOM); HTML elements are in a tree. The browser has a layout engine that figures out where everything goes, and a paint engine that draws it.
	1. ==Easy== to work with, but ==slow== for anything with thousands of moving parts -- every change triggers a recalculation of the layout tree.
2. A [[Canvas]] element, which is a blank bitmap (a rectangle of pixtels), which you can draw on using the Canvas 2D API:
	1. It's an **immediate mode** API: You tell it to draw a rectangle, and it draws pixels right now.
	2. There's no scene graph, no objects, no undo; ==if you want to animate, you *clear the canvas and redraw everything, every frame== (~60 times/second).*
	3. Canvas 2D runs on the ==CPU==; it's fine for charts and simple graphics, but ==not fast enough for millions of geometric features==.
```html
<canvas id="myCanvas" width="800" height="600"></canvas>
```
```javascript
const ctx = canvas.getContext('2d')
ctx.fillStyle = 'red'
ctx.fillRect(10, 10, 100, 100)
```
3. [[WebGL]], a JavascriptAPI that gives you direct access to your computer's GPU via the browser.
	1. You can use the same \<canvas\> element, but with a different context. The GPU can execute thousands of shader programs in parallel -- one per vertex, or one per pixel -- which is why it ==can render millions of map features at 60FPS==, while the CPU would choke.
	2. ==Extremely low level==! You hand the GPU raw buffers of numbers and tell it how to interpret them.
		1. ==No one writes raw WebGL for applications==; it's what ==libraries== like [[MapLibre GL JS]] and [[deck.gl]] are built on top of.
4. [[WebGPU]], the successor to WebGL, with a more modern API design. An emerging technology with better compute shader support. Not yet universally available, but [[deck.gl]] is starting to adopt it.









