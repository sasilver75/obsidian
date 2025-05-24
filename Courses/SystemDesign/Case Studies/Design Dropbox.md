SDIAH: https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox
SDIAH Video: https://www.youtube.com/watch?v=_UZ1ngy-kOI

Also referred to as "Design Google Drive"
- Very popular at Google, Amazon, Meta

I have some gripes with this video. I feel like handling what basically is a multi-leader write conflict scenario is something that wasn't really handled. As far as I know, this should basically be the same thing as a "Google Docs" thing; Different people are concurrently writing to their local, and those locals are intermittently .

---------

Before we start, let's remember our framework:
- ==**Functional Requirements (FRs)==**: "A user can ...."
- ==**Non-Functional Requirements (NFRs)==**: The qualities/"-ilities" of the system and other requirements
- (Optional: Back-of-the-Envelope (BOTE) calculations; try to defer to when you'll need them)
- **==Entities==**: List of the objects/tables, core entities that are persisted and exchanged via APIs
- **==API Interactions==**: Contract between client and our system
- (Optional for infrastructure interviews; Data Flow)
- ==**High-Level Design==**: Boxes and arrows; goal is to satisfy the FRs of the system
- **==Deep Dive==**: Expand on our high-level design and satisfy the NFRs of the system

-------

# 1) Functional Requirements
- If it's a system you know well, great. If it's something you don't know very well, this is where you'd ask your person a lot of questions.

Requirements:
- **Upload a file**
- **Download a file**
- **Automatically sync files across devices**
	- (You set up a local file on your desktop, and anytime a file is uploaded to remote storage, it's automatically synced with your local filesystem. Likewise if you drag and drop a file into your local storage, it should be uploaded to your remote storage and then subsequently synced with the other local devices on the account)

**Note:** This system design problem does not require you to design your own object/blob storage.


# 2) Non-Functional Requirements
- These will inform your deep-dives later on, and the depth of those deep dives is what's required for Senior+ ratings. So this matters!
- This is where we'd consider the [[CAP Theorem]], for instance:
	- Do we prioritize [[Consistency]] or [[Availability]]?
	- In this case, we're going to **prioritize Availability over consistency**
		- It's okay, if we upload or change a file in Germany, and someone goes in America to read that file shortly afterwards, and they see the older file for a number of seconds/minutes. It's more important that someone is actually able to download the file.
- NFRs:
	- **Availability > Consistency**
	- **Low latency upload and downloads** ("as low as possible")
	- Support large files, as large as 50GB (this is actually what Dropbox supports)
		- We should have **resumable uploads**; if they lose their connection halfway through a 50GB upload, we want to let them pick back up where they left off.
	- **High data integrity** is important (sync accuracy is high)
		- While eventual consistency is fine... when things have stabilized, the integrity of our data should be high!
		- What's in one folder should also be in another folder should also be what's in the remote storage.


**(SKIPPING BACK OF THE ESTIMATE ESTIMATIONS!)**
- Estimations are important, and you should use them... but you should use them in your high level design, when the result of your calculations actually has an impact on your design, which is hard to understand.
- It's not useful here to do some calculations, say "Wow, that's a ton," and then go about your day designing the same system that you would anyways. The interviewer only learns that I can do mulitplication.


# 3) Core Entities
- We don't know our columns, we don't know our schema, etc.
- But we can sit down and think about what core entities are exchanged in our system.

Entities
- **File** (raw bytes of the file)
- **File Metadata** (file name, id, mimetype, size, etc.)
- **Users** (in some cases, he recommends not even putting these; they're more a distraction than anything else)

Speaker recommends doing the columns for each of these **right next to the database** in the high-level design step.


# 4) API Endpoints

Let's make sure to have an endpoint(s) to satisfy each of our functional requirements:

FRs:
- **Upload a file**
- **Download a file**
- **Automatically sync files across devices**


So we'll need:
```

%% Create a file on remote %%
%% Note: It's likely that uploading a file IS NOT GOING TO BE A SINGLE REQUEST, since we're going to be doing multipart
uplaods. But its' fine AT THIS POINT for this to be the highlevel interaction! This is just a first guess. %%
POST /files -> 200
body: File and FileMetadata

%% Get a file from remote %%
GET /files/:id -> File + FileMetadata

%% Now we need to handle the synchonized stuff. %%
%% We imagine that our client needs to be able to notify the remote of changes on their local %%
%% As well as somehow get information on which files have changed on the remote %%

%% This is the one that gets a list of fileIds that have changed since a timestamp %%
%% We can take each of these fileIds and then call download on each of them, to get them locally. %%
GET /changes?since={timestamp} -> fileIds[]

```
Above:
- We're careful to not put "User" in the body for creating a file; we want to mention that the UserId is going to be present in the request's header via a [[JWT]].
- ((I'm interested  that he's not introducing some sort of patch endpoint for a file that we'd changed locally.))


# 5) High-Level Design
- Let's go one by one through our APIs and make sure that our system can satisfy the requirement and return the expected data.


- We're going to have a Client
- We're going to have an API Gateway that handles routing and cross-cutting concerns like Authentication, Rate Limiting, and SSL Termination
- We're going to have a File Service
	- The API Gateway will send an **upload(file, fileMetadata)** request to our File Service
	- We don't want to store each of these in the same database, because they're different in nature.
	- **==Oftentimes, we store raw bytes in blob storage, which is optimized for storing cheaply these large blobs of binary data.==**
	- So we're going to use [[Amazon S3]] to store the BlobStorage. This upload request from our File Service is going to return a FileUrl
		- Stoers Stores the raw bytes of the files
	- An then we're going to use some sort of File Metadata DB that we then write (after the file upload is complete) the file metadata to this DB. 
		- Table **FileMetadata**
			- fileid
			- file name
			- mimetype
			- size
			- ownerID FK to User
			- s3Link
			- ...any other metadata
![[Pasted image 20250524115653.png|800]]

Now, let's go to the next functional requirment:
**Get File given File ID**

- getFile(fileId) will hit our file service
- FileService will hit the FileMetaData DB to get the appropriate FileMetadata, then we go and get the File from S3, and we can return it to the user.
	- **In PRACTICE**: We'll get the file metadata, return the FileMetadata directly to the user, and then we can have the client download directly FROM S3, using the S3 link returned from the FileMetadata fetch.

![[Pasted image 20250524115854.png]]

Now let's focus on our last FR:
- **Automatically Sync Files Across Devices**

Notes
- Oftentimes we put the "Client" box on these SD interviews as this little small abstracted box. That's often appropriate, but for some systems, this being one (those others being those that require streaming/adaptive bitrates like Spotify, Youtube, these require logic on the client!)
- So let's reflect that by making the client bigger.
- Let's have a **Local Folder** on the client.
- Let's also require that users download some sort of **Client Application**. This client app is responsible for Syncing the local+remote foldres.
- So what does it take do this file syncing the simplest way?
	- Two syncing directions:
		- Remote changed
			- We want to be able too pull in those local changes.
			- We might poll for changes periodically, and then download the new file and replace our current one.
		- Local changed
			- If local changed, we need to be able to upload the changed files to remote.
- Let's talk about each of these
![[Pasted image 20250524120205.png]]
- **If Remote Changed**
	- Let's add a new service, called a **Sync Service**
		- (Aside: Some might argue that we're ==over-modularizing== our services here by separating these two)
	- We'll have a getChanges API for our **Sync Service**, which will ask the **FileService** to query its **FileMetadata DB** and tell us the ID of anything that's changed. Importantly, that means we want some more data on our FileMetadata!
		- Add CreatedAt
		- Add UpdatedAt
		- (this is why it's nice to have the definitions of the data next to the database in the high level design)
	- The sync service will respond with the IDs that are updated
	- The client app will then receive the IDs that have been updated
	- The client app will, for each of these:
		- request the FileMetadata for each of them, receiving the URL
		- request the URL, and download the video directly from S3.
	- Potentially, we don't want the SyncService to return just the MetaData, but also perhaps the full metadata for each of the files that have changed!
		- This would eliminate one of our requests in the above process, which is nice
		- We'd change our API definition
	- Q: How do we know that we don't already have that file in our local filesystem, when it changes on remote?
		- A: Good point! Our client app is going to need to have some LocalDB, which has the MetaData, including the ids and additional information about the files in the local filesystem!
		- When a new file updated and received locally, we need to look at the LocalDB to determine if we've already downloaded the File ID.
		- We'll save this for the Deep Dive.
- ![[Pasted image 20250524121456.png]]
- **If Local Changed**
	- If local changed, we want to upload those saved files to remote..
	- But how do we notice that?
	- Natively, each of these OS provides an API to watch changes to a folder or directory.
		- Windows: FileSystemWatcher is an API exposed by the OS, watches changes to a local directory, and can do something when a change is detected.
		- Mac: FSEvents is an object exposed which has a handful of endpoints, such as watching a local folder.
	- The point is: We'll have some code that watches for a change to occur, and when a change occurs, the client triggers some sort of upload via our normal **upload(file, fileMetadata) path**.
		- Maybe we re-upload the new file to S3, and then change the updatedAt time of the Flie Metadata
		- ((I don't know if I like reusing the POST for this ... PUT operation.))
![[Pasted image 20250524122334.png]]

# Deep Dive
- How deep you go here i largely a function of your seniority.
- If you're a mid-level candidate, what you have here is pretty good, and likely close to getting hired!
	- The interviewer might lead you to certain questions.
- If you're a senior or staff engineer, you might want to come over to your nonfunctional requirements, go through them one by one, and enhance your design to make sure that they're satisfied.
	- Senior: Go deep in 1-2 places
	- Staff: Go deep in 1-3 places

**Let's go deep... in more areas than are necessary for an actual interview**

So let's go through our NFRs in order:
- Availability >> Consistency
- Low Latency Upload and Downlload
- Support large files as large as 50GB
	- ResumableUploads
- High data integrity


Let's go out of order and start with ... 

#### Deep Dive: Support Large Files (50GB) and Resumable Uploads
- Let's talk about why our current design wouldn't likely work for large files, just files 5-10GB.
	- This design we have has two issues:
		- This is redundant! We upload the file once to the File Service, and one to BandWidth
			- It's a waste of bandwidth and CPU resources.
		- We likely couldn't even upload through this path if we wanted to!
			- **Request bodies have a limit!** This is set (differently) by any of Browsers, Servers, API Gateways, such that what can be in our Request body can only be so large.
			- AWS Managed Gateway has a limit of 10MB in a post body.
				- 50GB is is obviously far larger than that! We couldn't even get through there!

Let's start with the simpler **redundancy problem**
- We can actually have the client upload directly to S3, in the same way that we let the client download directly from S3
- But we don't want just ANY client to be able to upload to S3!
- When we upload a file...
	- The client uploads **just the metadata**, not the bytes of the file, to our **File Service**, which writes that data to the File Service DB, probably updating some new column called **Status**...
	- And instead of uploading the file to blob storage, we instead request a [[Presigned URL]].
		- We say: "We want to upload a file of this mimetype, it's this size, etc." And then Blob storage gives us a secure link, which has some additional metadata at the end of the URL, which is signed by S3, which ensures that the file uploaded is only of that mimetype and only of that size, and the link is only valid for some short window (e.g. 10-30 minutes)
	- We return this presigned URL to the client
	- The client can then upload directly to S3 with that presigned URL.

![[Pasted image 20250524122812.png]]
- That solves the first, issue, no problem!
	- ((they didn't talk about when we update the **Status** column on the FileMetadata database))

Now the second problem: **Request bodies having a limit, and the files being too big**
- If a user had a 100mbps upload speed and uploading 50GB, that would take a long time! The author quotes 1 hour 12 minutes, which I think is wrong.
- This would also suck from a UX perspective if I uploaded half of it and then my internet failed and I had to retry.
- We solve this by using **Chunking!**
	- We can chunk files on the client to 5MB chunks
	- On the client, we create these chunks. 
		- So we create 50GB/5MB = 10000 chunks.
		- So we can upload these chunks, and **keep track of the status of which chunks have been uploaded**. This is so if we fail we can compare the chunks we have uploaded to those in the file... and upload those chunks that are missing
	- So we need to update our FileMetadata db to store information on these chunks too...
		- (We haven't specified whether it's SQL or NoSQL; when well-configured, they can largely do the same things.)
		- If it were a DB like Postgres, we'd use another table like a Chunk table, but for this problem, we're going to pretend that we're using [[DynamoDB]], so we'll have a Chunks list on our FileMetadata,
			- Where each **Chunk** in the list has:
				- id
				- status
				- s3Link
	- So we:
		- Client Took our 50GB file
		- Client Chunked it into 5MB chunks
		- Client sent an upload request to our FileService
		- FileService gets a presigned url for each chunk, then writes the MetaData to the FileMetadata DB, and maybe the status of all these chunks are status=notStarted.
		- FileSErvice returns the FileMetadata to the client, which has presignedUrls for each of the 
		- The Client starts uploading these Chunks to S3 (either serially or in parallel)
		- As they get uploaded, we can update the status of each chunk to "uploaded", and upload the S3 link
			- ((how?))
		- If we get interrupted on the client, all we have to do is:
			- client requests teh FileMetadata
			- Receive the FileMetadata
			- Compare the chunks on the client to the chunks in the returned FileMetadata, and only upload the ones that are missing.
			- **But wait: how do we compare the chunks on our local to the chunks on the remote?**
			- We need some sort of unique way to be able to identify a chunk and track that chunk's identity!
			- We use something called [[Fingerprint|Fingerprinting]], which is just a hash over their bytes, giving us a unique ID.
			- This fingerprint is actually what we're going to store on our FileMetadata DB as our id for each chunk: the fingerprint
			- So we can compare the status of our fingerprints that match on ID
			- If there are any chunks that have differing statuses with their same-fingerprinted item, we'll re-upload them.

**One last thing we handwaved:**
- When the client is uploading chunks to S3, how do we update the status of our chunk in the FileMetadata databse?
	- Method 1: Rely on our client to make an additoinal request to our server (a patch, put) to update our FileMetadata to say "Complete Now!"
		- This relies on our client, and our client could lie to us. The client could tell us incorrect information, so that our status is inconsistent with what happened in S3.
			- "Who cares? hackers could only mess up their own files!"
			- But having inconsistencies between our FileMetadata DB and our S3 isn't ideal; it will cause headaches for the engineering team.
	- Method 2: We use a pattern called **==Trust but Verify==**
		- After the client gets a successful response from its chunk upload to S3, the client then makes a request to he FileService, which checks with Blob Service, and confirms that what the client is saying is true. If confiremd by S3, it then writes to FileMetadata service.
	- Method 3: Most blob storage cloud providers enable some version of [[Change Data Capture]]
		- Basically, you can have some notification triggered when blob changes happen.
		- S3 uses something called S3 notifications.
		- Whenever a new chunk is uploaded, it then can notify our File Service that a chunk has been updated.

**Also Note:**
- S3 enables an API called **==Multi-Part Upload==** that does this stuff we're talkign about: The uploading, fingerprinting, comparing of fingerprinting on the client to validate, etc. If you were to use this S3 MultiPart Upload, you'd run into a situation where on each chunk, S3 doesn't support a notification, so there's nuance there. 
- ==The **Trust but Verify** is the approach that the speaker would likely use in the interview.==


Questions:
- Do all chunks get different presigned url?
- We request a presigned url, then we stick it in the DB, and then return it to the user, right?
- How/when does our FileService (or FileMetadataDB) know to update the status of the chunks? The client is uploading those chunks directly to Blob Storage!
- Is there a problem with storing the presigned url in the filemetadat service? Do we just store the s3 url, and not the presigned suffix, but return the s3url+presigned suffix to the client?


#### Deep Dive: Low Latency Upload and Download
- **Chunking** of uploads is helpful here.
	- It helps us speed up the upload process
	- While bandwidth is fixed, chunking helps us make the most of the available bandwidth.
	- By sending multiple chunks in parallel, we can use adaptive chunk sizing based on network conditions, etc. to maximize the amount of bandwidth to use.
	- That helps us speed up our uploads a bit.
- We can talk about a [[Content Delivery Network]] if we wanted to...
	- CDNs are great to speed up the download process... but top candidates bring up a nice point when they talk about CDNs, where ==they weigh whether or not it's necessary in this case.==.
	- **Users are likely usually downloading their own files**, so they're likely located close to their respective servers (e.g. your local AWS datacenter) anyways. This might only be relevant if a given file is super popular (e.g. declaration of independence is in our dropbox, and people around the world are reading it.)
	- CDNs are pretty expensive to manage. Pros and cons, there's no right answer!
- To speed up, we can also **==Transfer fewer bytes over the network==** by using [[Compression]]!
	- By sending fewer bytes over the network, uploads and downloads can be quicker.
	- But if we compress things, we have to decompress things on the other side, which take a lot!
	- There are some filetypes that compress a lot, and some that don't
		- .txt files and .docx compress well
		- .jpg, .mov, .mp4 etc. are already naturally pretty well compressed, and so this will only gain us a few % in the size of the file, which might not be worth it given the added time that we need to do this.
		- So we can have some ==intelligent logic== on the client to determine whether we want to apply compression for a given file.
			- We'd have to  add some "CompresionAlgo" on our FileMetadat table to decompress respectively.

![[Pasted image 20250524130747.png]]

Questions:
- Wait, so how does the compression work again? The client may compress a file, and include in its upload(...) the compression algorithm used. That's written into the FileMetadata store.
- The client will later use a presigned url to upload the compressed file to blob storage
- Later, when (say a different) client wants to get that data, they get the filemetadata which includes the s3URL, then after they download teh compressed file using that url, they then decompress it on the client using the appropraite algorithm from the metadata?


## Deep Dive on Sync
- We want Sync to be fast...
	- When changes happen on remote, we'd like for them to happen on our local as soon as possible.
	- We want our sync to be consistent, meaning whatever is in our local folder is in remote, and whatever is in our remote folder is local.
- So how do we make sync fast?
	- How do we know when there are changes?
		- The client can periodically [[Poll]] for changes, using that SyncService
			- We can change the polling rate based on what the client is doing (==adapative polling==); maybe if the client is doing a lot of work, we increase  the polling rate, etc. Also provide a "Refresh" button to the client which polls for them.
			- This is appropriate
		- Clients sometimes bring up [[Websockets]] or [[Long Polling]], but it's totally overkill
			- You don't want to set up a persistent, long-lived connection between client and server if you don't have to -- it introduces a bunch of overhead with a WebSocket manager, etc.
			- Form a product requirement perspective, do you care about **milliseconds** vs **20 seconds**?
			- Long Polling also doesn't make a lot of sense... it's for when you're expecting responses in a relatively quick window, but in this situation we don't know when updates are going to happen.
- If a change got made to a file remotely... **==Do we want to have to download the entire file to update it locally, if only one part of it was changed?==**
	- This is where **Chunking** helps again! We only get the chunks that have changed!
	- So we probably want to get an updatedAt
	- So when we poll for changes, we just want to look for the chunks that have changed since some provided timestamp.
	- So we'll add to our Chunk the information **updatedAT**
		- So we only need to download the chunks that have been updated since our last polling.
- We're going to do some adaptive polling, and also this delta sync to only check changed chunks.
![[Pasted image 20250524132151.png]]

Question:
- This still seems like it's leaving nuance on the table..
- Like say we have a document that's a cover letter, and we have 3 chunks associated with it, and each chunk "is' a paragraph in the document.
- So we upload that document for the first time from client to remote, and we create our MetaData object with chunks 1-3, and their respective fingerprints and updatedAt.
- What happens when the client (or another) then changes the middle chunk of the three chunks?
	- That chunk's fingerprint is going to change.
	- We need both:
		- A way to delete the old chunk 
		- A way to create the new chunk
	- and a way for clients to understand when they later fetch a document that this change has occurred.
		- getChanges(), on our current one.
- Q: Do chunks need some sort of ordering to them? How does this get complicated?


## Deep Dive: Consistency
- We keep referring to Polling, but we've intentionally been a little vague about what that means....
- Oh, we'll probably also need a **FolderId** on our FileMetadata
- So we'll say: **"For folderID 123, get me any files for which a chunk has changed or been added since the last sync**"

Dropbox does something differently, which is having an EventBus (e.g. Kafka) with a cursor...
- Any change that occurs to any give FileMetadata, we pout that change event in each bus
- We'll need some additional state
	- ==Folder==
		- Cursor (sync cursor)
As events or changes are coming in, we put those changes on our EventBus

The very first time you set up a local folder, you won't have any Cursor

We'll replay all of the local events to construct the correct local state on the client.
We'll update the cursor to say "This is where you were in the update process"
The next time that we want to sync... instead of going to the DB to ask for changes ,we go down to the event bus, and there's a linear list of changes... we navigate to the cursor (the last event we read), and we take all the new events that have come in, and we apply those change to our local DB.

==This is definitely more complicated ((and not very well explained))==

While this is what Dropbox does, for this design and our FRs, this is overklil
- Event Bus with a cursor is great when you nee auditing/version control, so that you can roll back/forward events, but we didn't have any of those in our functional requirements.
	- BUT IN YOUR INTERVIEW, THIS MIGHT BE IMPORTANT!


![[Pasted image 20250524133727.png]]


It's possible that our local and remote fall into an inconsistent state. 
We need to apply [[Reconciliation]]
- Our client app will periodically go fetch all the information in remote for this current folder
- It will make sure that everything is totally consistent, and if there are any issues, it will handle those issues.






![[Pasted image 20250524134002.png]]
**==HERE ARE OUR FINAL UPDATED API ROUTES, BY THE WAY!==**



































