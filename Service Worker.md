A browser-run JavaScript worker that sits between a web page and the network. It lets a web app perform more like an installed one, especially around offline behavior and network caching.

It can intercept network requests from the page and decide how to respond:
- Go to the network
- Return something from the [[Cache API]]
- Fall back to an offline page
- Update cached assets in the background

Commonly used for:
- Offline supports for [[Progressive Web Application]] (PWAs)
- Caching static assets like JS/CSS/images/fonts
- Caching API responses
- Background sync
- Push notifications

Important traits:
- Runs separately from the main page thread
- Cannot directly access the [[Document Object Model|DOM]]
- Uses asynchronous APIs
- Requires [[HTTPS]], except on `localhost`
- Has a lifecycle: install, activate, fetch, update

