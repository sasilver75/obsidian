
302 vs 301


`301 Moved Permanently`: Redirect, permanent; cached
- "The URL of the requested resource has been changed permanently. The new URL is given in the response."
- Does reduce load on our actual servers.

`302 Found`: Redirect, temporary; not cached
- "This response code means that the URI of the requested resource has been changed *temporarily.* Further changes in the URI might be made in the future, so that the same URI should be used by the client in future requests."
- In a case where you have analytics, and if you want to show in a dashboard (in a UrlShortening problem) how often people are requesting shortUrls, then this might be important.
- ==Typically, you want to go with this, because it allows you to understand if things are working.==

`307 Temporary Redirect`
- "The server sends this response to direct the client to get the requested resource at another URI with the same HTTP method that was used in the prior request. This has the same semantics as `302`, with the exception that the user agent *must not change* the HTTP method used (if a POST was used in the first request, a POST must be used in the redirected request)."

`308 Permanent Redirect`
- "This means that the resource is now permanently located at another URI, specified by the `Location` response header. This has the same semantics as the `301 Moved Permanently` HTTP response code, with the exception that the user agent *must not* change the HTTP method used."