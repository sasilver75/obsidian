Began: 2008
Link: [CommonCrawl.org](https://commoncrawl.org/)

Common Crawl is a nonprofit organization with an eponymous dataset of ==petabytes of **monthly** web-crawls==, collected ==since 2007==. Contains raw web page data, metadata extracts, text extracts. 
Releases a new crawl containing ==200 to 400 TiB of textual content== obtained via automatic web crawl ==every 1-2 months==.

Example: The latest CC crawl (April 2024) contains 2.7 billion webpages, with 386 TiB of uncompressed HTML text content.
96 crawls have been released since 2013, and 3 crawls from 2008-2012, which were in a different, older format.


CommonCrawl data is available in two main formats: 
- ==WARC== (Web ARChive format): Contain the raw crawl data, including full-page HTML and request metadata.
- ==WET== (WARC Encapsulated Text): A text-only version of those websites
The [[FineWeb]] 🍷 people from 🤗 say that while many dataset creators use the WET files as their starting point, in their experience the extraction used by CommonCrawl to create these WET files is suboptimal *for the goals of LLM pretraining*, and a variety of open-source libraries provide better extraction; they use the [[Trafilatura]] library.
![[Pasted image 20240605133601.png|200]]






Problems/Challenges:
- They're a nonprofit and only cover a fraction of the web.
	- Therefore our LMs aren't trained on all the web
- Biased sampling towards the US
- Lots of data issues; labels for some languages that CC has might be totally off; the Arabic issue - many pages tagged as Arabic aren't even Arabic. Pages that are too big, deleted, duplicated are ignored. SPAs are ignored; CC doesn't render JS, so much of FB isn't under CC (this is a problem re: the "walled gardens" of the internet (Discord, Slack, FB))