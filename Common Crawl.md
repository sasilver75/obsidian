Began: 2008
Link: [CommonCrawl.org](https://commoncrawl.org/)




Common Crawl is a nonprofit organization with an eponymous dataset of ==petabytes of **monthly** web-crawls==, collected ==since 2008==. Contains raw web page data, metadata extracts, text extracts. ==250+ billion webpages== spanning 17 years, cited in over 10,000 research papers.


Founded by Gil Albaz
- He started the predecessor to AdSense and sold it to Google in the late 90s/early 00s. He saw how important crawling was for Google. Basically quit Google and started CommonCrawl to enable Google competitors.


Issues:
- They're a nonprofit and only cover a fraction of the web.
	- Therefore our LMs aren't trained on all the web
- Biased sampling towards the US
- Lots of data issues; labels for some languages that CC has might be totally off; the Arabic issue - many pages tagged as Arabic aren't even Arabic. Pages that are too big, deleted, duplicated are ignored. SPAs are ignored; CC doesn't render JS, so much of FB isn't under CC (this is a problem re: the "walled gardens" of the internet (Discord, Slack, FB))