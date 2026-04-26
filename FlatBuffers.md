A Google Serialization library.

Comparison with [[Protobuf|Protocol Buffer]]:

  Protocol Buffers (Protobuf)
  - Schema-defined binary serialization
  - Designed for network messages and RPC (it's what gRPC uses)
  - To read a field, you parse the whole message sequentially
  - Smaller wire format, but requires parsing/copying data into native objects before
  use

  FlatBuffers
  - Also Google, but different team, different goals
  - Designed for zero-copy access — you read data directly from the buffer without
  parsing it into intermediate objects
  - Access a field by computing its offset and reading bytes directly
  - Faster reads at the cost of slightly larger size
  - Designed for game engines and performance-critical systems (originally built for
  games at Google)

  The key difference:
  Protobuf: bytes → parse → native object → access fields
  FlatBuffers: bytes → access fields directly (no parse step)

  FlatBuffers is better when you have large binary data and only need to read a subset
  of fields — you don't pay to deserialize fields you don't access. This is exactly why
   FlatGeobuf uses it — when doing a spatial query you're skipping most features
  entirely, so zero-copy access to just the fields you need (bbox, geometry bytes) is a
   significant win.


![[Pasted image 20260425202855.png]]