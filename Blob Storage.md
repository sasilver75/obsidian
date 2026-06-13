---
aliases:
  - Object Storage
---
Highly scalable and replicated ways of storing immutable file data.
- "BLOB" = "Binary Large Object"

Pros: Typically cheaper for storing static data than using a distributed file system.
Cons: To do processing on the file data, we must first load it into a distributed file system, then run a batch job. ((?))


### How do authorized reads from Object Storage work, in an application?

Typically, it's one of:
1. Streaming the object through your application
2. Issuing a short-lived signed URL.
3. Issuing a short-lived CDN signed URL or signed cookie.
4. Using cloud-native identity and access management for trusted client.

The common default for web applications is: private object storage + application authorization check + short-lived signed download URL.

For most web applications:
1. Store files in private object storage
2. Store file metadata and ownership in your application database
3. Have user request downloads from your application by file id
4. Authenticate the user
5. Authorize access against your database
6. Generate a short-lived signed URL
7. Redirect the user's browser to that signed URL
If you *really need strict control,* you can proxy the serving of content through your actual application.
If you *really need high-scale delivery,* you can use a CDN + signed URLs or signed cookies.


Otherwise, patterns include:
##### (1/4) Streaming the object through your application  ((Atypical))
- The client asks your application for the file, your application authorizes them, retrieves the object bytes itself from the application, and returns them to the client.
```
Browser -> App: GET /downloads/invoice-123
App: authenticate user
App: authorize user for invoice-123
App -> Object Storage: GET private object
Object Storage -> App: object bytes
App -> Browser: object bytes
```
- In this model, the browser never talks to directly to object storage. This is useful when you need very tight control:
	- Hiding object keys completely
	- Check authorization continuously
	- Log every byte served through your application.
	- Apply transformations/watermarking/virus scanning/auditing.
	- Revoke access immediately.
- The downside is your application now incurs the network bandwidth + memory burden, which, for large files or high traffic, can be expensive and less scalable.

##### (2/4) Issuing a short-lived signed URL. ((==Most Common==))
- The client first asks your application for permission to download the file, and after authn/authz, your application fetches and returns a [[Presigned URL]], which the client can use to directly download the object bytes.
```
Browser -> App: I want file invoice-123.pdf
App: authenticate user
App: authorize user
App -> Browser: signed URL valid for 60 seconds
Browser -> Object Storage: GET signed URL
Object Storage: validate signature and expiration
Object Storage -> Browser: object bytes
```
- The [[Presigned URL]] contains authorization data, usually including:
	- The object key
	- The allowed HTTP method (e.g. `GET`)
	- An expiration timestamp
	- A cryptographic signature
	- Sometimes response headers such as `Content-Disposition`
- The Presigned Url is a *bearer capability:* whoever possesses the URL can use it until it expires. The object store doesn't ask "Is Alice allowed to download `invoice_123`?" Instead, it asks: "Is this signed URL unexpired and valid, according to a trusted credential."
```
https://example-private-bucket.s3.us-east-1.amazonaws.com/reports/2026/invoice-123.pdf
	?X-Amz-Algorithm=AWS4-HMAC-SHA256
	&X-Amz-Credential=ASIAIOSFODNN7EXAMPLE%2F20260613%2Fus-east-1%2Fs3%2Faws4_request
	&X-Amz-Date=20260613T184512Z
	&X-Amz-Expires=300
	&X-Amz-SignedHeaders=host
	&X-Amz-Security-Token=IQoJb3JpZ2luX2VjE...very-long-url-encoded-session-token...
	&response-content-disposition=attachment%3B%20filename%3D%22invoice-123.pdf%22
	&X-Amz-Signature=4f3b7a9c2e1d0f8a...64-hex-characters...
```
Above:
- See that it uses AWS Signature Version 4 with [[Hash-based Message Authentication Code|HMAC]] using [[SHA-256]]
- See that we have an access key + a credential scope (access-key-id/date/region/service/aws4_request). The access key ID is visible, but the secret access key is not in the URL.
- We have the UTC timestamp when the URL was signed.
- We have an expiry, showing that it expires 300 seconds (5 minutes) after the signing date.
- We have inclusion of HTTP headers in the signature; for download, usually only `host` is signed
- The security token appears when the signer used temporary credentials, like an [[Amazon Security Token Service|AWS Security Token Service]] assumed role.
- The response content disposition tells S3 to return a `Content-Disposition` response header, so the browser downloads file with a friendly filename.
- The signature at the end is the actual cryptographic proof. After user request of the URL, S3 will recompute the signature from the HTTP method, object path, query parameters, signed headers, timestamp, region, and credential scope. If anything signed changes, the signature no longer matches.

> A presigned URL is a normal object URL plus a temporary, URL-encoded proof that someone with valid storage credentials authorized this exact request.

##### (3/4) Issuing a short-lived [[Content Delivery Network|CDN]] signed URL or signed cookie. ((Relevant for giving access to many files, or for very large files where you want to limit travel))
For large-scale downloads, a content delivery network can sit in front of private object storage.
```
Browser -> App: request download
App: authenticate and authorize user
App -> Browser: CDN signed URL or signed cookie
Browser -> CDN: request file
CDN -> Object Storage: fetch private object if cache miss
CDN -> Browser: file bytes
```
- This is common for: Paid video content, software downloads, large media files, static assets requiring authorization, global users where latency matters.
- There are two common CDN variants:
	- Signed URL: Best for one specific file download
	- Signed cookie: Best for access to many files under a path, such as a video playlist and its segment files.
- More on signed cookies:
	- A video streaming service might need signed cookie because a single video might requier many segment requests. Signing each segment URL separately is possible, but signed cookies are often simpler.
```
/course-7/video/segment-001.ts
/course-7/video/segment-002.ts
/course-7/video/segment-003.ts
```
For [[Amazon CloudFront]], the signed cookie might look like:
```http
HTTP/1.1 204 No Content
Set-Cookie: CloudFront-Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9tZWRpYS5leGFtcGxlLmNvbS9jb3Vyc2VzL2NvdXJzZS03Lyo...; Domain=media.example.com; Path=/courses/course-7/; Secure; HttpOnly; SameSite=Lax; Max-Age=300
Set-Cookie: CloudFront-Signature=GJX9L6ZC1Wq9X7...long-url-safe-signature...M6RyQ__; Domain=media.example.com; Path=/courses/course-7/; Secure; HttpOnly; SameSite=Lax; Max-Age=300
Set-Cookie: CloudFront-Key-Pair-Id=K2JCJMDEHXQW5F; Domain=media.example.com; Path=/courses/course-7/; Secure; HttpOnly; SameSite=Lax; Max-Age=300
```
Above, the Cloudfront-Signature is a cryptographic signature over the policy, so that CloudFront can verify that the policy was signed by a trusted private key.
Above, the Cloudfront-Key-Pair tells CloudFront which public key or key group to use when verifying the signature.
Above, the CloudFront-Policy has the [[Base64]]-encoded policy document, which conceptually looks like:
```json
{
  "Statement": [
    {
      "Resource": "https://media.example.com/courses/course-7/*",
      "Condition": {
        "DateLessThan": {
          "AWS:EpochTime": 1781373600
        }
      }
    }
  ]
}
```
Above, this just says the allow access to resources under this path until a certain timestamp.

Then the browser can request matching CDN URLs normally, and the cookies are sent along with it:
```http
GET /courses/course-7/video/segment-001.ts HTTP/1.1
Host: media.example.com
Cookie: CloudFront-Policy=eyJTdGF0ZW1lbnQiOlt7...;
        CloudFront-Signature=GJX9L6ZC1Wq9X7...;
        CloudFront-Key-Pair-Id=K2JCJMDEHXQW5F
```

##### (4/4) Using cloud-native identity and access management for trusted client.
- Sometimes the client itself has a cloud identity or temporary cloud credential:
```
Client -> Identity Provider: authenticate
Identity Provider -> Client: temporary cloud credential
Client -> Object Storage: GET object using credential
```
- This is most common for internal tools, server-to-server systems, mobile apps with carefully-scoped temporary credentials, and enterprise environments using federated identity. 
- This is less common for ordinary public web applications, because exposing cloud-style credentials to browsers is easy to get wrong.




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




