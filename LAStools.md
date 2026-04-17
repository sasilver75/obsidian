Proprietary software by Martin Isenburg (also creator of [[LAZ]] compression). ==The industry standard for production LiDAR processing== -- faster than anything else for large datasets and extremely well-optmized.

Key tools:
- `lasground`: ground classification
- `lasheight`: height above ground
- `lasclassify`: building and vegetation classification
- `las2dem`: interpolate [[Digital Elevation Model|DEM]] from ground points
- `lasnoise`: noise filtering
- `lastile`: tile large datasets
- `lasmerge`: merge multiple files
- `laszip`: compress/decompress LAZ

The ==licensing is complicated== ; some tools are free for non-commercial use, others require paid licenses. For open source pipelines, people often work around it with [[Point Data Abstraction Library|PDAL]].




