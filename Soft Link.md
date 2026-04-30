---
aliases:
  - Symlink
  - Symbolic Link
---









______________

# How the UNIX Filesystem Stores Files
- A file has two separate parts on disk:
	- The ==[[Index Node|inode]]==  (index node), a small data structure that ==holds everything about a file *except its name and contents==*
		- File size, permissions, Owner UID and group GID, Timestamps, ==pointers to the disk blocks where the actual bytes live==, a reference count (link count) of how many names point to this inode.
		- Each inode has a unique number in its filesystem (e.g. 8429471) which you can see with `ls -i <file>`.
	- The actual ==directory== entry. A special file, whose contents are a table of (name, inode_number) pairs:
		- For Directory: /home/sam/
			- hello.txt -> 8429471
			- notes.md -> 8429471
			- src -> 8429473 (this inode is a directory itself)
- When you type `cat hello.txt`, the OS:
	- Looks up `hello.txt` in the current directory, gets inode `842971`
	- Reads the inode and finds the data blocks.
	- Returns the bytes
- ==So the filename is just a label in a lookup table.== The "real" file is the inode and its data blocks.

Once you understand this, links are a little more obvious -- they're basically just other entries in that lookup table.