---
aliases:
  - CNAME
---
A [[Domain Name Service|DNS]] record type that marks one domain name an *alias* for another domain name.
- CNAME stands for "Canonical Name"

```
www.example.com  CNAME  app.hosting-provider.com
```
Means:
- When someone asks for `www.example.com`, treat it as another name for `app.hosting-provider.come`.
- Note that the CNAME record doesn't point to an IP address, it points to another DNS name, and then that target name is resolved separately.

Use a CNAME when you want one name to follow another name's DNS configuration:
- `www.example.com` -> `example-host.vercel.app`
- `blog.example.com` -> `myblog.medium.com`
- `status.example.com` -> `statuspage.vendor.com`
- `docs.example.com` -> `docs-hosting-provider.com`

This is somewhat common for SaaS platforms, hosting providers, documentation sites, status pages ,and CDNs.


# A CNAME is NOT an HTTP [[Redirect]]. 
- A CNAME like
```
www.example.com CNAME app.host.com
```
- This ==DOES NOT tell the browser the change the URL== to `app.host.com`!
- The browser still shows: `www.example.com`
- DNS only helps the client find where to connect. HTTP redirects happen later, at the web server or application layer.


