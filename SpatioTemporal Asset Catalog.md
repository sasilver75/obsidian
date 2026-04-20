---
aliases:
  - STAC
tags:
---
References:
- [CNG Video: Introduction to STAC](https://www.youtube.com/watch?v=1KPvEnH29sU)
_________

A JSON metadata standard for discovering [[Raster]] datasets.
- Tool to create Catalogs

Commonly paired with [[Cloud-Optimized GeoTIFF]]s (COGs); ==STAC+COG is the standard stack for satellite imagery.==
- STAC: Tells you what exists and where the files are
- COG: Lets you efficient read just the spatial subset you need

An open specification for describing geospatial assets in a consistent, searchable way.
Problem it solves: Every satellite data provider had their own metadata format and discovery mechanism, and STAC standardizes both:

Four components:
1. Item: Describes one asset (one Landsat scene, one Sentinel-2 tile; a [[GeoJSON]] feature).
2. Collection: Groups related items (all Sentinel-2 L2A scenes).
3. Catalog: A top-level container linking to collections.
4. API: Standard REST search endpoint; filter by bbox, datetime, cloud cover, collections.


![[Pasted image 20260419225513.png]]

![[Pasted image 20260419225548.png]]
![[Pasted image 20260419225709.png]]
Above: If you want to query only the building footprints in Austin, you can do that from within [[DuckDB]], here reading from an [[Amazon S3|S3]] URL that [[Overture Maps Foundation|Overture]] maps hosts. This is without needing a ton of money/compute.

![[Pasted image 20260419225657.png]]
Same thing with [[Zarr]], requesting 2 meter resolution temperature, with the data streaming into the bucket every hour or so, I think.

![[Pasted image 20260419225747.png]]
![[Pasted image 20260419225922.png]]
Planetary compute stack API, looking [[Sentinel|Sentinel-2]] data with a bounding box over Austin in a certain date range, looking for the best image with a certain cloud coverage.
- This returns the full satellite image intersecting your [[Area of Interest|AOI]], and you still need to clip it.
He made a false color visualization where he swapped out the blue band with the [[Near Infrared|NIR]] band, which is good for vegetation analysis... [[Normalized Difference Vegetation Index|NDVI]] is the near-infrared minus the red band, normalized; bright green is healthy vegetation.

![[Pasted image 20260419230142.png]]
[[National Agriculture Imagery Program]] (NAIP) flies the US with really high resolution (~1m, sub-meter) every two years or so.



_________________

STAC is really two things
- stac-spec: Metadata language in JSON for describing geospatial data, and comes with Extensions that can go off of it.
- stac-api-spec: Searching and accessing that data through an APiI, can have Extensions

So "STAC" refers to these two specifications.
![[Pasted image 20260420114601.png]]

Static vs Dynamic Catalogs
- Above is a a static catalogue: static files on disk/blob storage.
	- Have a root catalog, which contains links to other sub-catalogues, other catalogues in the structure. This is for partitioning collections and items in different ways: we might have a bunch of [[Moderate Resolution Imaging Spectroradiometer|MODIS]] collections in one sub-catalogue, or we might want to divide and partition our items up by year, and we could do that.
	- The point is that the items are linked together, so we can crawl the catalogue from the root down to a useful atomic unit, often a scene, which have individual data files called assets.

![[Pasted image 20260420115047.png]]
A Dynamic catalogue (or an API) is a little differnet because there's no reason to have subcatalogues because you can search data in any way you want...
- For the vast majority of use cases, peple aren't working with static catalogues, they're working with APIs.
- Root catalog is API landing page
- Underneath, we have collections, which have items, which have assets.
- ![[Pasted image 20260420115136.png]]
- (still dynamic)

KEy points of STAC:
- Built on standards (JSON, GeoJSON, OGC APIs)
- Simple and stable core structure
- Metadata decoupled from data, which could be somewhere else entirely
- Supported by an Open-Source ecosystem


![[Pasted image 20260420115459.png]]
Main atomic unit for you to care about as a user is an item
- An Item is a [[GeoJSON]] object with some additional STAC fields
	- stac_version:
	- links:
	- assets:
	- stac_extensions (maybe):
	- collection (if it's part of a collection):
![[Pasted image 20260420115702.png]]
- Links for an item
	- You'll see a self link, a parent link, and a collection link if it's in a collection, and a root link. These are the ==hierarchical links== used to traverse a catalogue.
		- An items parent is a collection, a collection's parent is the root (in an API/dynamic catelogue)
	- A canoncial link and via link... ==relational-type links.==.. are more best-practifces that WE (speaker) do. When we have an API, we don't treat it as the golden copy of metadata, we treat it as an index.. the original copy of that metadata is alongside the data in blob storage. So canonical... means that it's the same metadata record, but in blob storage, which is realiable/redundant , and not relying on a service to provide copy of metadata. If anything happens ot index, we can reindex pretty easily. Via... means it was created from an XML file.
	- These are what you see in the Sentinel API in EarthSearch, but there could be a bunch of links. We encourage to use many links to docs, software, etc. But you'll alalyays have those hierachical leaks.

![[Pasted image 20260420120125.png]]
- These are your basic properties: Platform, constellation, etc.

![[Pasted image 20260420120139.png]]
Assets:
- Most importantly, an href to where the data is located, the type of data (COG), title, name...
- Roles is important: there are roles for assets, since asset don't necessarily need to be just data; you could have an originalm etadata asset hat will have a role of metadata.

![[Pasted image 20260420120326.png]]
- This is a visual asset containing 3 bands. This is describing metadata per band; see that it has "name"?, like we had in assets? So name can be dsecribed at asset level or band level to descri be teh 3 bands in a single asset.
- Might have to specify metadata for every band in an asset, for instance. Think of these as discrete variables contained in a file.

![[Pasted image 20260420120400.png]]
What about extensions?
- The ones tha you should know about are whichever ones are in the data that you're using!
- In the item, there should be a stack extension list.


![[Pasted image 20260420120526.png]]
You might want things that are less than 10 degrees off-nadir, for intsance.

Name can also be defined at the item level, under item properties, if it applies to the entire item!
If it applies to only one asset of several assets, you can define it at the asset level.
If it only applies to a band within an asset, you define it at the band level.



As a user, what software should I use?
![[Pasted image 20260420120803.png]]
- pysatc-client is for searching APIs and getting the objects back
- odc-stac lets you load them into an XAraray and now you can use the python ecosystem to do things with them.
- There's a whole github repo called stac-utils in which there's a whole bunch of tools! You can look through this and see whate else might be useful to you.

STAC for providers:
- See ![[Pasted image 20260420120901.png]]
- Read this document! It has guidelines on how to name things, etc.
- Consider how the data is going to be used by the user. Teh user is going to be interseti n a collection, so don't mix data where users are only interested in a part of it, etc.
	- For instance, don't put level 1 and level 2 data in the same collection. It's allowable, but it's not a great idea.
- As a provider, probably use `Stac-fastapi` or `Stac-server`, probably not `Pygeoapi`, which isn't as feature rich 
	- Stac-fastapi is what's used by [[Microsoft Planetary Computer|Planetary Computer]], so has a lot of resources put into it. The main backend is a PostGIS backend.
	- Stac-server is used by [[Element84 Earth Search|Element84]], which is a Node server using [[ElasticSearch]] backend; ElasticSearch is great at going spatial aggregations.
- Should I use `stactools`? For Landsat, Sentinel 2, etc... the answer is NO. Don't use stactools; they're great examples of how STAC items can be constructed for various datasets, but it's grown to a messy thing that's unmaintained. Best going back to using pystac or riostac to create basic items, etc... even though you might come across information about using stactools to create your definitions.. that's outdated information.
- Typically you want uniform assets for the items in a collection.
	- item_assets are defined at the collection level; they're not required, but it lets a user look at a collection nad see what assets are expected when they look inside a collection.
	- ![[Pasted image 20260420121114.png]]

What extensions should I use?
![[Pasted image 20260420121151.png]]


STAC Workflows
- STAC isn't just for consuming data; it's not just for finding data, etc.
- It can also be used as the central messaging between steps in workflows.
- ![[Pasted image 20260420121602.png]]
- This is an example of some files that a scientist speaker is working with looked like
- People will create these files... and will use the filename as a way of trying to store metadata (eg the datetime in the filename). Don't try to store metadata in the filename!  
- Instead of thinking of your input/output as files, think:![[Pasted image 20260420121700.png]]
- Metadata in, metadata out. 
- Your workflow/function/etc takes in one or more STAC items, which it can (if it decides to) download. It can augment that metadata with additional metadata, it can add an asset, etc.
	- If you have a workflow that creates a cloud mask, you inputa STAC item, create a cloud mask, put that up in the cloud, add that asset, and return the new STAC
- ![[Pasted image 20260420121810.png]]



Challenges in STAC
![[Pasted image 20260420121854.png]]
- We saw this sheer scale of metadata problem: NASA CMR is the metadata repository for NASA data... and it has billions of granules/items in it. And it's JSON. So that's a problem. People working on it.
- Flexibility being the enemy of interoperability:
	- STAC is flexible, you can do whatever you want. People don't often read the best practices, and do things in different ways. Different STAC APIs (even mirrors of the same dataset) aren't completely interchangeable, along with the proliferation of extensions.
- Shoehorning problems into STAC
- STAC governance via steering committee was a bit of a problem for a while, originally steering committe was basically all north-american males working for private companies, which didn't have a lot of diversity.
	- ![[Pasted image 20260420122200.png]]
	- This is the new one

![[Pasted image 20260420122213.png]]
Community meetings on Mondays @ 11a every two weeks.
