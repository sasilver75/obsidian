The simplest [[Geospatial Index]] to understand and implement, which is why it's the default choice in many databases.
The core idea is simple:
- ==Convert a 2D location into a 1D string in a way that preserves proximity.==


![[Pasted image 20250520182448.png]]
- Above: A Geohash is essentially this process of recursively dividing the world into smaller squares, with each division adding more precision to our location description.
- A [[Geohash]] is essentially this process, but using a [[Base32]] encoding that creates strings like "dr5ru" for locations. ==The longer the string, the more precise the location!==
	- dr = San Francisco
	- dr5 = Mission District
	- dr5ru = Specific city block

==The useful bit is that locations that are close to eachother usually share a similar prefix string!==
- Two restaurants on the same block might have geohashes that start with `dr5ru`

==Once we've converted our 2D locations into these ordered strings, we can use a regular [[B-Tree]] index to handle our spatial queries!==
- Remember the B-Trees can excel at **prefix-matching** and **range queries?** That's exactly what we need in proximity searches!
- Finding nearby locations becomes as simple as finding strings with matching prefixes
	- ==Looking for restaurants near geohash "dr5ru" means we can do a range scan in our B-Tree for entries between dr5ru and dr5ru~, where ~ is the highest character.==

This is why Redi's geospatial commands use this approach internally!
When you run 
```redis
GEOADD restaurants 37.7749 -122.4194 "Restaurant A"
GEORADIUS restaurants -122.4194 37.7749 5 mi
```
Redis is using a Geohash under the hood to efficiently find nearby points.
- Redis's `GEORADIUS` and other implementations like this handle the complexity of doing radius queries where, if we're looking near "dr5ru", we'd query for locations starting with "dr5ru" ,"dr5rv", "dr5rt", etc. ==Turning radius searches into the appropriate set of prefix queries + distance filtering isn't always trivial to get right, and it's nice to have a tool do it for you==.

**==WARN:==** The ==main limitation of geohash== is that locations near each other in reality might not share similar prefixes if they happen to fall on different sides of a major grid division - like two restaurants on opposite sides of a street that marks a geohash boundary. But for most applications, this edge case isn't significant enough to matter.

==The elegance and simplicity of turning a complex 2D problem into a simple string prefix matching problem that lets us leverage existing B-Tree implementations are why Geohashes are such a popular choice.==




