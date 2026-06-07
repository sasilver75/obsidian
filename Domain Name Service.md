---
aliases:
  - DNS
---
Translates human-readable domain names (`google.com`) into [[Internet Protocol|IP]] addresses (`142.250.x.x`) so that your browser knows where to "find" a website.

So how does the resolution work?
1. You type `google.com` into your Browser
2. Browser checks its own DNS [[Cache]]. If not found, continue.
3. Your [[Operating System]] checks its own local cache / `/etc/hosts` file. If not found, continue.
4. Your machine asks a DNS [[Recursive Resolver]], which might be run by your [[Specific Impulse|ISP]], or [[Cloudflare]]'s DNS (`1.1.1.1`) or Google's DNS (`8.8.8.8`), etc, asking: "What is the [[A Record]]/[[AAAA Record]] for `google.com`?"
5. The resolver checks its own cache. If not found, it has to resolve the name by "walking" the DNS hierarchy.
6. The recursive resolver asks a replica of one of 13 DNS [[Root Nameserver]] identities, asking: "Where do I find information about `.com` domains?" The root server doesn't know Google's IP, but replies with the nameservers for the `.com` [[Top-Level Domain]].
7. The resolver then asks a `.com` [[TLD Nameserver]]: "Where do I find information about `google.com`?" The `.com` server doesn't usually know Google's final IP either. it replies with Google's [[Authoritative Nameserver]]s.
8. The resolver ask's Google' authoritative nameserver "Where is the IP address for google.com?", and the authoritative nameserver replies with the actual [[Domain Name Service|DNS]] records (e.g. [[A Record]])! `google.com -> IP address(es)`. Often times there are multiple IPs for load balancing/geography.
9. Resolver returns the answer to your computer's OS/browser, and also caches the answer.
10. Your browser can now request that specific IP address.

Then normal web connection steps happen.
1. [[Domain Name Service|DNS]] might have returned multiple [[Internet Protocol|IP]] addresses; the browser/OS chooses one.
2. The client opens a connection. For normal [[HTTPS]], this is first a [[Transport Control Protocol|TCP Handshake]]: `SYN`, `SYN-ACK`, `ACK`. Now we have a reliable byte stream between client and server!
3. The [[Transport Layer Security|TLS Handshake]] starts, because we're using `https://`. The browser negotiates encryption before sending the HTTP request. The browser sends a ClientHello, including things like its supported cipher suites.
4. The server proves its identity with a [[Certificate]] for `google.com`, the browser verifies that the certificate is for for `google.com`, it was signed by a trusted [[Certificate Authority]] (CA), the cert is not expired, and the certificate chain is valid. 
5. Using the TLS handshake, client/server agree on [[Symmetric Key Encryption|Symmetric Encryption]] keys. After this point, HTTP data is encrypted.
6. Browser sends the HTTP request:
```
GET / HTTP/2
Host: google.com
User-Agent: ...
Accept: text/html, ...
Cookie: ...
```
7. Google's edge/[[Load Balancing|Load Balancer]] receives it. That IP probably doesn't represent an application server, it likely reaches an edge proxy/load balancer which may do: ([[TLS Termination]], route based on hostname/path, check cookies/session, apply security rules, load balance to backend services, serve cached/static content)
8. Application backend generates or retrieves a response
9. Server sends HTTP response, routed back through any intermediate Google-internal infrastructure
```
HTTP/2 200 OK
Content-Type: text/html
Cache-Control: ...
Set-Cookie: ...

<Response Body>
```
10. Browser parses the HTML response and renders. If the page references more resources (e.g. CSS, JavaScript, images, fronts, API calls), then the browser fetches those too. Each resource might need its own DNS/connection setup/TLS/HTTP, unless an existing connection can be reused.




