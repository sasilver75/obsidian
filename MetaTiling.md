[[Cloud-Optimized GeoTIFF|COG]]s are row-major, meaning blocks in the COG in the same row are stored adjacent to eachother on disk.
If you have a chunk size that's not good for your application, you need to physically rechunk your data... especially for temporal chunks (eg daily, monthly, yearly - chunking can matter).

With MetaTiling, ==you can logically change chunk size without changing the physical chunk size==.

