---
aliases:
  - Lossy Compression
  - Lossless Compression
  - External Compression
  - Internal Compresion
---
An algorithm that makes data smaller, at the cost of having to encode data into the compressed format before saving, and having to decode data out of the compressed format before usage.
- In most cases, the benefits of smaller file sizes when stored outweigh the time it takes to encode and decode the compressed format.
- Can either be external or internal, and either lossy or lossless


# Lossy vs Lossless Compression
- ==Lossy Compression==: A type of compression where the exact original values CANNOT  be recovered after decompression, meaning the compression process permanently loses information. Tend to give smaller file sizes than Lossless Compression. 
	- Examples: [[Joint Photographic Experts Group|JPEG]], LERC, WebP, AVIF, MP3, AV1
	- Use for images, audio, and video where approximate reconstruction is acceptable.
- ==Lossless Compression==: A type of compression where the exact original values **can** be recovered after decompression. This means that the compression process does not lose any information. Lossless compression codecs tend to give larger file sizes than lossy compression codecs. 
	- Examples: [[gzip]], deflate, LZW, [[Zstandard|zstd]], Snappy, LZ4, Botli
	- Use this for API responses, logs, database records, backups, and messages.


# External vs Internal Compression
- ==External Compression==: Compression not part of a file format's own specification, which is added on after the main file has been saved. Tends to be used as part of [[ZIP]] archives or with standalone [[gzip]] compression. Typically makes a file no longer [[Cloud-Native Geospatial|Cloud-Optimized]].
- ==Internal Compression==: Compression that's part of the file format's own specification. Formats like [[Cloud-Optimized GeoTIFF|COG]], [[Cloud-Optimized Point Cloud|COPC]], and [[GeoParquet]] include internal compression; useful for [[Cloud-Native Geospatial|Cloud-Optimized]] data formats, because it allows internal chunks to still be fetched with [[HTTP Range Request]]s, but have smaller sizes.
	- Typically, for files that have already been internally compressed, adding another layer of external compression ([[gzip]]) will not make the file smaller, and will reduce performance by requiring an extra decompression step before the data can be used.


# Uses in a System Design context
- Usually used to trade CPU time (compressing, uncompressing data) for lower network bandwidth and storage space, and sometimes better throughput.
- ==Compression does not magically make a system faster.== Compression makes a system faster *only when the saved I/O time is larger than the added CPU time*. If CPU is already saturated, compression can make latency and throughput worse.

| Use case                 | Why compression helps                 | Common examples                                              |
| ------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| API responses            | Reduces bytes sent over the network   | HTTP gzip, Brotli, Zstandard                                 |
| Static assets            | Speeds up page loads                  | JavaScript, CSS, HTML, SVG compressed by CDN                 |
| Object storage           | Lowers storage cost                   | Compressed JSON exports, backups, archives                   |
| Logs and observability   | Logs are repetitive and compress well | gzip/Zstandard-compressed log files                          |
| Message queues           | Increases effective broker throughput | Kafka batch compression with Snappy, LZ4, Zstandard          |
| Databases                | Reduces disk I/O and storage usage    | Page compression, column compression, dictionary encoding    |
| Caches                   | Fits more items in memory             | Compressed Redis or application-cache values                 |
| Backups and snapshots    | Saves storage and transfer time       | Compressed database dumps or filesystem snapshots            |
| Cross-region replication | Reduces expensive WAN traffic         | Compressed write-ahead logs or change data capture streams   |
| Analytics systems        | Makes large scans cheaper             | Parquet/ORC compression, run-length encoding, delta encoding |
| Mobile/offline sync      | Saves battery and bandwidth           | Compressed sync payloads                                     |
| Media delivery           | Makes large assets practical to serve | JPEG, WebP, AVIF, H.264, AV1, Opus                           |

