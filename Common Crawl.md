Began: 2008
Link: [CommonCrawl.org](https://commoncrawl.org/)

Common Crawl is a nonprofit organization with an eponymous dataset of ==petabytes of **monthly** web-crawls==, collected ==since 2007==. Contains raw web page data, metadata extracts, text extracts. 
Releases a new crawl containing ==200 to 400 TiB of textual content== obtained via automatic web crawl ==every 1-2 months==.

Example: The latest CC crawl (April 2024) contains 2.7 billion webpages, with 386 TiB of uncompressed HTML text content.
96 crawls have been released since 2013, and 3 crawls from 2008-2012, which were in a different, older format.





Problems/Challenges:
- They're a nonprofit and only cover a fraction of the web.
	- Therefore our LMs aren't trained on all the web
- Biased sampling towards the US
- Lots of data issues; labels for some languages that CC has might be totally off; the Arabic issue - many pages tagged as Arabic aren't even Arabic. Pages that are too big, deleted, duplicated are ignored. SPAs are ignored; CC doesn't render JS, so much of FB isn't under CC (this is a problem re: the "walled gardens" of the internet (Discord, Slack, FB))