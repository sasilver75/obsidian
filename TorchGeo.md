References:
- [Video: Robin Cole: TorchGeo 1.0 with Adam Stewart](https://www.youtube.com/watch?v=0HfykJa-foE&t=107s)


[[PyTorch]]'s official geospatial deep learning library from [[Microsoft Research|MSR]], analogous to torchvision but built for [[Remote Sensing]] and [[Remote Sensing|Earth Observation]] tasks.
> "It's the same thing as TorchVision, but it's for satellite imagery."

Raw geospatial data is hostile to standard ML pipelines:
- Imagery has arbitrary [[Coordinate Reference System]]s, resolution, and extent: You can't just stack tensors.
- Multi-sensor fusion requires spatial alignment across datasets.
- [[Remote Sensing|EO]] datasets have non-standard splits (spatial autocorrelation breaks random train/test splits)

Provides:
- Datasets
	- 50+ ready-to-use EO datasets ([[Sentinel]], [[Landsat]], NAIP, BigEarthNet, etc.)
	- Handles download, checksum, CRS normalization automatically.
- Samplers
	- Geospatial-aware sampling: Random geographic patches, tiles by grid, etc.
	- Respects spatial boundaries to avoid data leakage between train/test splits.
- Transforms:
	- Handles arbitrary band counts (not just RGB), e.g. 13-band Sentinel-2
	- Spectral index computation ([[Normalized Difference Vegetation Index|NDVI]], etc.)
- Pretrained Models:
	- Hosts pretrained weights for EO models like [[Clay]] and other foundation models.


![[Pasted image 20260420104012.png]]
From: https://www.youtube.com/watch?v=HPlZbkNa6zI 
![[Pasted image 20260420104127.png]]
If you don't do overlapping patches you get strange edge/boundary effects, typically does worse at the edge.
These things (see the video above)... are what separates the kids from the adults.


__________________

[Video: Robin Cole: TorchGeo 1.0 with Adam Stewart](https://www.youtube.com/watch?v=0HfykJa-foE&t=107s) August 2025

TorchGeo is all of the high-level abstractions specific to Geospatial data:
- Data loaders, foundation models, transforms, and [[PyTorch Lightning]] modules for using these things in combination.
- Enables you to do things like: Load datasets, train models... etc.


TorchGeo is designed for things like Multispectral or [[Hyperspectral]] images, which don't have 3 bands, so [[Torchvision]] transforms don't really work. It's hyper focused on these kinds of "obscure" image datasets, where you have to deal with things like reprojection and automatically aligning vector/raster datasets, rasterizing a vector, etc.

> "I think it's the most popular geospatial deep learning library... it's been quite successful, much more so than any of us expected."


We just announced the TorchGeo Organization.
- When TorchGeo started 4 years ago as his intern project at the AI4GOOD research lab at Microsoft, they made it open source, thinking it might be useful, and were surprised by adoption.
- With the competition over who has the best foundation models, best software libraries, etc... We wanted to make TorchGeo even more open! It's already open-sourced under an [[MIT License]] but also wanted to have open and independent governacne so that it no longer belongs to Microsoft, it belongs to you, the TorchGeo contributors!
	- To set up this governance panel was actually Microsoft's idea! 
- There's the TorchGeo Organization, which governs the TorchGeo Project.
	- The Organization is led by the technical steering community:
		- Adam Stewart + Niels from Tomb (?)
		- Caleb and Anthony from Microsoft
		- Isaac from Wherobots
		- Ashlyn from Space42

Torch Geo itself, the Python library isn't hte only thing we manage, there's a HugginFAce and Zenoto (?) community that we update models.

He's managing a few masters students that are managing integration for TorchGeo: Imaging loading up QGIS or ArcGIS and chatting with a vision-language foundation model and using it to label your satellite imagery for you!
- That's about a year out, but would also be something that the committe would govern.

We do want to grow this:
	- How can I take part? Other libraries built on top of TorchGeo especially... who want to have a say in backwards-incompatible changes or release..


Coming up on TorchGeo 1.0 release so we can guarantee some stability
- The one major feature we're still missing is complete Time Series support, which is something that's lacking in almost all GeoML libraries at the moment.
- We tried to take inspiration from SITS (CITS? It's an R library) to see how we could integrate this into it.

There's One-Diemsnional (1D) time-series support (an air pollution sensor, where you want to forecast future levels)
For 2D, we've always done that in TorchGeo
But we also want to do 3D: A location's been damaged by some natural disaster or warfare, so you have two timestamps for one location (before and after) and you want ot label damaged buildings, etc.
We want to enable multiple timestamps, where we have an entire year's worth of sentinel imagery, and we want to take advantage of both the time dimension and the spatial one
And there's the 4D space: Targeting medium-range weather forecasting, decade-scale climate monitoring... you have X, Y, Z and time as well. 
All of these are covered under time series support, and we chose these all to highlight the fact that Torch is not only for satellite imagery; we just merged our first weather forecasting model, we have a number of datasets for point data, and a lot of other interesting applications.

[[R-Tree]] lacks some of the features that we need:
- We want to be able to separrately index space OR time, not both at the same time.
- WE did a long and intensive software literature reivew
- We look at Shapely's SDR Tree, GeoPandas, STAC ecosystem to find something a little more flexible and powerful... and we ended up switching to GeoPandas, so I had to rewrite every dataset to work with GeoPandas, lol.

We're looking at hte DAtloaders, how they load dataloaders, etc. We used [[rasterio]], which was great, becaues it wraps [[Geospatial Data Abstraction Library|GDAL]], but there are a lot of things that we nwat to read that GDAL itself can't read; A lot of climate data is stored in [[Network Common Data Form|NetCDF]], [[Hierarchical Data Format 5|HDF5]], and techniclaly GDAL can read these things, but not the verison that's on [[PyPI]]... we started looking at [[Xarray]] (and specifically [[rioxarray]], a library that addsa  .rio accessor to xarray DataArray and Dataset objects, bringing rasterio/GDAL-backed geospatial capabilities into the xarray world!.)
- We're also planning on replacing [[Fiona]] with [[Geopandas]] so that we can dorp some of the indirect dependencies...
- Things like Geopanadas and rioxarray have good integrations with [[Dask]] so you can parallelize things, and they scale up very well!

So this is why we wanted to have a breaking 1.0 release.

Everything related to GeoDataset and GeoSampler is likely to break, the rest of the library won't change that much.

- Just merged a new ChangeDetectionTrainer so that you can do binary, multi-class and multi-label change detection with TorchGeo...
- Also working on an AutoRegressionTrainer which you can plug in for things like air pollution forecasting, weather forecasting, etc.

It's partly about features, but also about making sure that the foundations are there...

He talked to some of the folks at Pytorch Lightning, and their thing was:
- Get to the point where it's stable, and then move on to the next big thing!
- We want to add [[SpatioTemporal Asset Catalog|STAC]] integration, etc... We want to get it to he point where it's relatively stable, where Adam as the main developer can step back and say "I want the community to do a lot of work," and I can explore new things (eg GIS integrations to low-code/no-code environments for non-GIS experts.)

"TorchGEO is hte best library ever for a ML expert, but for a RS expert, there are a lot of things that re hard to wrap your ehad around: What's a DataLoader, Sampler, etc."
- It's obviously much more complex than a simple [[Scikit-Learn]] workflow.


Q: You mentioned that Torch is built on by a few other libraries, which?
A:
- Terratorch
- GeoAI
- GeoBench benchmarking suite uses TorchGeo to curate many of its datasets
- NASA's `pytorch-caney` uses TorchGeo for dataloading
- 


Q: What'st your take on foundation models? 
A: I'm biased since I'm a foundation model researcher... and have been working on some of those for a while...
Did a lot of stuff on ResNEt, Vision TRansformer backbones, different SSL techniques ... something like 40 foundation models for LandSAT; researchers said: "What's the point of 40 foundation models?"... Ya lol
A: When I joined TUMunich; we have a lot of foundational models for Sentinel 1 and 2, we have some for Landsat... there are still thousands of satellites out there that we aren't covering. we want foundation models for every satellite, not just the most popular ones. When I joined this lab, DOFA ([[Dynamic One for All Model]]), where the idea was ... you can use a dynamic hypernetwork to take hte weavelength of each spectral band as input and dynamically generate hte first layer of your vision tranfsormer, so as long as you know hte wavlenghts of your spectral bands, a signle set of pretrained weights works for every earth observation satellite ever launced, and that will ever be launcehd in the future! It's a 

