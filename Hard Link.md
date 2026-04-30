

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


# Hard Links
- Just another directory entry pointing to the same inode:
	- Directory: /hom/sam
		- hello.txt -> 8429471           <------ These are called "Dentry"s or directory entries.
		- greeting -> 8429471     <---- hard link, same inode
	- Both names refer to the same file; there's no "original" and no "copy," they're equally valid names the same inode. This inode's **link count** is now 2.

Example:
```
$ `ln hello.txt greeting`
$ `ls -li`

8429471 -rw-r--r-- 2 sam staff 13 Apr 30 hello.txt
8429471 -rw-r--r-- 2 sam staff 13 Apr 30 greeting
	    ↑                       ↑
     same inode             link count = 2
     

Aside:
  8429471  -rw-r--r--  2  sam  staff  13  Apr 30  hello.txt
     ↑          ↑      ↑   ↑     ↑     ↑     ↑        ↑
   inode    perms    link owner group size  mtime  name
   number          count
```

Properties of hard links:
- Edit one, see the change in the other (they're the same bytes)
- Delete one, the other still works
- Data is freed only when the link count hits zero.
	- This is how `rm` works for all files! There's no separate "delete" operation, just "remove a name and decrement the count"
- Identical metadata: Same size, permissions, timestamps, because there's only one inode.
- ==Cannot cross filesystems==; Inode numbrs are scoped to a single filesystem.
- Typically cannot link to directories; allowing this could create cycles in the directory tree.

What [[pnpm]] uses this for:
```
~/.pnpm-store/v3/files/.../react-18.2.0/index.js inode: 8429471 
~/projects/app-a/node_modules/react/index.js inode: 8429471 (hard link)
~/projects/app-b/node_modules/react/index.js inode: 8429471 (hard link)
```


# Soft Link (Symbolic Links)
- Completely different from a Soft link! Instead, Soft Links are a small file whose ==contents== are a ==path string== pointing at another file:
	- Directory: /home/sam
		- hello.txt -> 8429471 (regular file)
		- greeting -> 8429999 (symlink, its OWN inode)
	- Above, Inode `8429999` is a special "symlink" inode whose data is literally the string `hello.txt` or `/home/sam/hello.txt`.
Example:
```bash
$ ln -s hello.txt greeting       # the -s means soft/sym
$ ls -li                   

8429471 -rw-r--r-- 1 sam staff 13 Apr 30 hello.txt
8429999 lrwxr-xr-x 1 sam staff  9 Apr 30 greeting -> hello.txt   # The leading "l" means "symlink"
	 ↑                                            ↑
 different inode                            stored path
```

When you `cat greeting`, the OS:
1. Looks up `greeting` and gets `inode 8429999`
2. Reads the inode and sees its a symlink, contents are `hello.txt`
3. Resolves `hello.txt` *relative to the symlink's directory*
4. Looks up `hello.txt` in that directory, gets inode 8429471
5. Reads the data

This per-access resolution is why symlinks have very different properties than hard links.

Properties of soft links:
- ==Can cross filesystems==: They store a path string, not an inode reference
- ==Can point to directories==: common and useful
- Can point to nothing (if you delete the target, the symlink still exists but is now "broken", "dangling", reading it gives `ENOENT`)
- Can point to relative paths: `ln -s ../config.json` link is resolved at access time, so moving the parent directory may or may not break it depending on whether the relative path still resolves.
- Has its own metadata; permissions on the symlink itself usually don't matter, most syscalls "follow" the symlink and apply to the target. 
- Can chain; A symlink can point to asymlink and can point toa srymlink.

Package managers like `pnpm` use symlinks *on top of hard links*  to construct each project's `node_modules` tree:

```
node_modules/react              → symlink to → ../.pnpm/react@18.2.0/node_modules/react
                                                                                ↑
                                                                    this directory contains
                                                                    hard-linked files
```
The symlink lets pnpm assemble a per-project dependency tree without copying or hard-linking the directroy structuer itself; only the leaf files are hard-linked.

![[Pasted image 20260430103144.png]]


# The Mental Model to Carry Away
  - A ==hard link== is another name for the same file. There is no "original."
  - A ==soft link== is a tiny file containing a path that gets resolved on every access. It's a pointer, and
  pointers can dangle.
  - Filenames and files are separate things. Once you see the directory-entry-vs-inode split, every weird
  link behavior makes sense.


# The full pnpm Storage Chain

![[Pasted image 20260430103353.png]]
- ==Global store (~/.local/share/pnpm/store)== — one per machine. Holds the actual file bytes, named by content
  hash. Every pnpm project on your machine shares this. Survives rm -rf node_modules.
- ==Project .pnpm/ directory== — one per project. A flat list of every package version your project uses, each
in its own isolated directory. The files here are hard links into the global store.
- ==Project node_modules==/ — what Node.js's resolver looks at. Contains symlinks pointing into .pnpm/.

Why two layers?
- The symlinks in the top-level `node_modules` exist because Node's module resolver walkes `node_modules/pkg` and doesn't know about `.pnpm/`, so `pnpm` gives it Node what it expects (a node_modules/react path) while keeping the actual storage flat and deduplicated underneath.