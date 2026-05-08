---
aliases:
  - APNs
---
An Apple-operated relay for push notifications for iOS apps.

You (app server) don't talk to the iOS application directly, you talk to APNs and APNs talk to devices.
- When the app launches and the user grants permission, iOS registers with APNs and hands the app a unique token (per device, per install)
- The app POSTs that token to your server, which stores it on the user row. 
	- Tokens can rotate (reinstall, restore-from-backup), so the app re-uploads on every launch.
- Your app server, when it wants to notify a user, looks up the user's tokens and sends an [[HTTP 2]] request to APNs with the payload + token. APNs authenticates our app server using a `.p8` signing key or a `.p12` cert tied to your Apple Developer Team ID.
	- These `.p8` signing keys are generated in the Apple developer program, tied to our Team ID.
- User permission: The app calls `UNUserNotificationCenter.requestAuthorization(...)`... iOS shows the system prompt. If the user denies, you get nothing - there is no override. ==This is why apps usually ask at a contextually-relevant moment, not at first launch!==




