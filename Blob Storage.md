---
aliases:
  - Object Storage
---
Highly scalable and replicated ways of storing immutable file data.

Pros: Typically cheaper for storing static data than using a distributed file system.
Cons: to do processing on the file data, we must first load it into a distributed file system, then run a batch job.

e.g. [[Amazon S3]]



___________

[Object Storage in System Design Interviews w/ Ex-Meta Staff Engineer](https://youtu.be/RvaMHMxHjp4?si=ldZQSSpD_JgBfAgD)

Basically a database designed for Binary Large Objects (BLOBs), things like videos, photos, music files, JSON files, large text files... Each of these is just large collections of bytes.

Q: So why don't we just store this stuff in normal databases? (e.g. Relational OLTP)
A: Those databases are built for small, frequently-changing records that might require joins, etc. They're not build for large, mostly static files.
![[Pasted image 20260606201842.png]]
Imagine you were storing your actual profile pictures alongside your users, for instance.
- [[PostgreSQL|Postgres]] packs all of these rows internally into 8KB [[Page]]s.
- If you were going to have a 4MB image, that would span 500 whole Pages for a single row!
	- Why is this bad?
	- What if we had a query where we wanted to pull in the top 50 users?
	- ==Overhead problem==: This would create a lot of extra overhead! The database has to manage and access many more piles. This impacts performance, increased memory pressure, nad slow down what should be fast, performant queries.
	- ==Replication problem:== When you replicate data across database servers, that 4MB blob has to be copied to each replica. This increases the amount of 
	- ==Backup problem==: These 4MB images will be included in your (e.g.) nightly backup snapshots that you run.
		- This turns a restoration process (which should take minutes) into something that takes many long hours.

So traditional relational databases choke on BLOBs, in terms of performance and cost.
So we should be storing these in our object/blob storage.


So how does Object Storage work?
![[Pasted image 20260606202147.png]]
The key is that all of your files are stored on a cheap storage node in a data warehouse somewhere.
- How do we know where our file is?
- When a client wants to view a file, they make a request to a metadata service, which is part of object storage.
- This metadata service uses some sort of index; it determines that Server A holds file 1 (which is actually stored on many servers for redundancy).
- Server A's responsibility is to stream the bytes of this file back to the client
So it's a simple lookup, a connection, and a stream of the file back to the user.
((==I think he's missing a step...== I'm pretty sure that (for *private* bucket access) the initial request from the client would return a [[Presigned URL]], and then the client would make a second GET to that presigned URL which would result in the streaming of bytes from the storage node.))


What makes this so cheap and durable in the first place?
1. Flat namespaces = quick file looksup
	1. Unlike your local machine where you have to navigate through folder trees to find your filed, object storage uses a flat structure (these "folders" are just UI cosmetic sugar to make it easier for us humans to use. In reality, under the hood, it's just using a single string, which lets us locate just the files we need.)
2. Immutable writes = no locks or race conditions
	1. In OLTP databases, you're modifying things in storage, etc. 
	2. In object storage, you can't modify bytes in an object; you can only:
		1. Overwrite the file entirely
		2. Create a new version
3. Redundancy: "11 nines" of durability
	1. Every object is fully replicated or erasure-coded across multiple different servers/racks/datacenters.


# What you should know for an interview?

![[Pasted image 20260606212244.png]]
Large files should be stored in Object Storage, but you need to store the metadata (e.g. the linkToObjectStore, photoURL, etc.) in your transactional database.

In (e.g.) Postgres, you store a stable object reference in your entity table, and generate presigned urls (see below) when needed.
Common column names:
- object_key, storage_key, s3_key, blob_key

==Typically, we store the object key, and keep the actual bucket name in config/environment variables==.

- So we would store, in our `s3_key` column `users/123/avatar/01JZQM37K9V6B2N4P8R.jpg`
- And we would know that the bucket is `AVATAR_BUCKET=user-uploads` from our config file
- And then we could (when we need to) combine these two into an S3 [[Uniform Resource Identifier|URI]] : `s3://user-uploads/users/123/avatar/01JZ8Q3M7K9V6B2N4P8R.jpg`
	- Or as an HTTPS [[Uniform Resource Locator|URL]]: `https://user-uploads.s3.amazonaws.com/users/123/avatar/01JZ8Q3M7K9V6B2N4P8R.jpg`

(Note: The URI format is typically used for SDKs/CLIs/internal references/logs/config/database storage, while the HTTP URL tells an HTTP client/browser how to request an  object over the web.)

And then we could get S3 to stick a presigned URL on top of it at runtime when the user needs upload/download access:

`https://user-uploads.s3.amazonaws.com/users/123/avatar/01JZ8Q3M7K9V6B2N4P8R.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA...%2F20260607%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260607T190000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=abc123...`





![[Pasted image 20260606212435.png]]
You can both upload and download from S3 using [[Presigned URL]]s.
- In the upload case (e.g.):
	- You might think that you'd upload the file to your server, and that your server would then upload to object storage with the correct  permissions. ==NO!== That uses a lot of bandwidth; if your server isn't well-scaled, it shouldn't be handling things like influx of large video files. 
	- Instead, you have your server request from the object store a [[Presigned URL]] which your client can use to [[HTTP]] PUT that file directly to the object store for the next (e.g.) 5 minutes.


![[Pasted image 20260606212648.png]]
Multi-Part upload lets us upload parts of a document in chunks, and then stitch them together.
- Important: ==On the internet, there are limits on the size of a file that can be POSTed/PUT on an internet==
	- These limitations are in browsers, in gateways, in the servers themselves, etc. They exist all over.
- In S3's case... their limit is ((Currently in June 2026, 50TB)) for a blob
	- So our client can take the file, chunk it into a bunch of chunks, upload each of those chunks in parallel, and then our object store stitches those chunks together. If a chunk upload fails, it can be retried.




