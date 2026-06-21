e.g. Dropbox

> Dropbox is a cloud-based file storage service that allows users to store and share files. It provides a secure and reliable way to store and access files from anywhere, on any device.

### Functional Requirements
1. A user can ==upload== a file from any device.
2. A user can ==download== a file from any device.
3. Users should be able to ==share== a file with other users, and view files shared with them.
4. Users can automatically ==sync== files across devices.

### Non-Functional Requirements
1. The system should be highly ==available== (prioritizing availability over consistency; it's not google docs, it's dropbox)
2. The system should support files as large as ==50GB==
3. The system should be secure and reliable. We should be able to recover files if they are lost or corrupted.
4. The system should make upload, download, and sync times as ==fast== as possible.


### Core Entities
- It seems that we have a few nouns flying around:
	- File: The raw data that users will be uploading/downloading
	- FileMetadata: The metadata associated with the file, including information like the file's name/size/mime type, and user who uploaded it.
	- User: The user of our system.
- 


### Data Flow
- 


### High Level
- 


### Low Level
- 

