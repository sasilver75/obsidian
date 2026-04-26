The process of ==identifying which pixels in a satellite image are covered== by ==clouds== (or ==cloud shadows==) and marking them as invalid, so they aren't used in analysis.
- Clouds are opaque to visible and near-infrared wavelength; a pixel covered by cloud is measuring the cloud, not the ground, and ==using it as if it were a valid ground observation will corrupt any analysis==.
- Most satellite imagery products ship with a pre-computed cloud mask.



Clouds
- Clouds are bright (high reflectance across visible bands) and cold (low thermal infrared brightness temperature); simple thresholds on these properties catch most clouds.

Cloud Shadows
- Harder than clouds themselves; they're dark and easy to confuse with water or dark vegetation, and their position depends on sun angle and cloud height geometry. Most masking algorithms handle them separately.




![[Pasted image 20260425230635.png]]