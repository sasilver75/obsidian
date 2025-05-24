SDIAH: https://www.hellointerview.com/learn/system-design/problem-breakdowns/dropbox
SDIAH Video: https://www.youtube.com/watch?v=_UZ1ngy-kOI

Also referred to as "Design Google Drive"
- Very popular at Google, Amazon, Meta

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

Speaker recommends doing the columns for each of thsee **right next to the database** in the high-level design step.


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
	- So we're going to use [[Amazon S3]] to store the BlobStorage. This upload request from our File Service is going to return 































