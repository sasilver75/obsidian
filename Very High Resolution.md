---
aliases:
  - VHR
---
References:
- [Audio: Chris and Krishna talk VHR imagery](https://christopherren.substack.com/p/krishna-talks-high-resolution-imagery)

A general term for very high resolution imagery, typically sub-meter.
- [[Maxar]]
- [[Planet Labs|Planet]] [[Planet Labs|SkySat]]

Data volume increases quadratically. Typically images smaller [[Swath]], but still have data volume issues.

They do agile pointing, tilting and rotating (slewing) based on what they're tasked to image by their customers, generating a lot of [[Nadir|Off-Nadir]] imaging.
- If you're [[Mosaic|Mosaicking]] or doing time-series, you have to... do processing to align pixels for things that are imaged. 
- There's a lot of positional error of satellites themselves.
- You might get a meter or two of pixel alignment all the time.
	- With 10m pixels, it sucks, but it's kind of fine.
	- With 50cm pixels, that means... that if you have 5m positioning error, you have 10 pixels of misalignment relative to the ground.
- Trying to do time series analysis from 5 [[Planet Labs|SkySat]] acquisitions... they do some level of [[Orthoimage|Orthorectification]], but they're optimizing for the global, not for your local area... they don't do that for you.
- So you have 5 images, and you have 5-10 pixel shifts per timestep easily, and basically ... ==all the time-series stuff that you can do with the grosser [[Sentinel|Sentinel-2]]... kind of doesn't work well because of all of these imaging issues that you get with VHR imagery!==
- Because of the agile pointing, you're not imaging at the same time every day... might be imagine off-nadir, and between acquisitions you might have 30 degree off-nadir from this direction, the next time 25 degrees in a different direction, etc.

People mostly aren't doing time-series analysis with Hi-Res though, they're mostly doing image interpretation and visual inspection-type tasks.
- Maybe you're looking at port activity for ships or parking lots, etc, where the positioning above doesn't matter ("Whatever, I'm just looking for the things I'm looking for!")





Even still, you almost never get full coverage over (e.g.) Gaza for a full day, will have to [[Mosaic]] from a full week.

