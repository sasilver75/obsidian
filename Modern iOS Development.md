
The biggest shock coming from cloud/backend is that Apple owns the distribution pipeline end-to-end.
- There is no `apt install your-app`. 
- Every binary is code-signed against an Apple-issued certificate, gets reviewed by Apple before reaching users, and runs in a sandbox Apple defines.
- ==There is no rollback button.== If you ship a broken update, you can pull the build from sale, but every user who already updated has the broken version. 
	- The fix is to ==ship a new version through review== (~1 day expedited, at best). 
	- This is why beta testing through [[TestFlight]] is genuinely load-bearing, not optional theater.

## The Lifecycle of iOS Apps
1. ==Setup== (one-time): You need a Mac; you install [[Xcode]] from the Mac App Store, and you pay Apple $99/year for an ==Apple Developer Program== membership. Without it, you can only distribute to yourself.
2. ==Scaffold==: In Xcode, "New Project" -> iOS App -> SwiftUI. You get a folder with a `.xcodeproj` file (Xcode's project metadata), some Swift source files, an `Info.plist` (declarative app config, version, permission strings, etc.) and an `Assets.xcassets` (images, colors, app icons).
3. ==Local dev loop==: You write Swift, hit CMD+R, and Xcode compiles and launches the ==iOS Simulator==, a full iOS instance running on your Mac. Hot reload exists (==SwiftUI Previews==), but is flaky; mostly, you do edits, refresh with CMD+R, and wait 5-30 seconds to see result. No physical device needed at this stage.
4. ==Run a real iPhone==: Plug in a phone, register it as a "==development device==" in Apple's portal, Xcode generates a provisioning profile (a signed file linking your Apple cert + your app's bundle ID + the device's UDID). Hit CMD+R, and the app [[Sideload]]s onto the phone for 7 days before it expires.
5. ==Beta distribution==: You upload a build to [[App Store Connect]], Apple's web portal for managing your apps. From there, [[TestFlight]] distributes it to up to 10,000 external testers via email invite. TestFlight builds get a lightweight Apple review ==(~24h review, sometimes minutes==).
6. ==Production release==: From App Store Connect, you submit a build for full ==App Review==. A human reviewer (plus automated checks) inspects it for full policy compliance, privacy disclosures, crashing, broken links, etc. ==Typically 1-3 days review, but sometimes 1 week.== Rejections are common and require a back-and-forth message thread with the reviewer. Once approved, you can release immediately or schedule it.
7. ==Updates==: Some loop: bump version, archive build, upload, review, release. There's a **==phased release==** feature (auto-rollout to 1% -> 100% over 7 days) for de-risking.
8. ==Watching it in the wild==: App Store Connect gives you crash reports, basic install analytics, ratings, and reviews. For deeper observability, you bolt on third-party tools ([[Sentry]], [[PostHog]], etc.).

# Languages and UI
- [[Swift]]: Apple's language. Strongly typed, value semantics by default (structs everywhere). What you'll write 99% of the time. Modern Swift as async/await and structured concurrency that'll feel familiar from TS/Python.
- Objective-C: The previous-generation language, still underlies many Apple frameworks but you'll rarely write it for new code. Swift can call into it transparently.
- [[SwiftUI]]: Declarative UI framework, conceptually similar to [[React]]. State drives view; you describe what the UI looks like for a given state. Modern, fast to develop in, but with a few year of "doesn't quite do that yet" potholes.
- [[UIKit]]: The older, imperative framework. Powers most apps shipped in the last 15 years. Mature, complete, verbose. [[SwiftUI]] is built ON TOP of it, and you'll occasionally drop down to UIKit for things SwiftUI can't do.

# Tooling
- [[Xcode]]: The IDE. Mac-only, Apple-controlled. Editor + compiler + debugger + simulator + Interface Builder + asset catalog manager + profiler launcher all in one. Monolithic and opinionated.
- Xcode Command Line Tools: CLI build/test commands (`xcodebuild`, `xcrun`, etc.). This is what [[Continuous Integration|CI]] uses. You can build without launching the IDE.
- ==iOS Simulator==: Runs a near-real iOS on your Mac. Great for UI iteration. ==Doesn't simulate camera, GPS movement, push notifications well.==
- ==Instruments==: Xcode's profiler. CPU, memory, energy, network, GPU, animation hitches, etc.
- SwiftUI Previews: Live preview of SwiftUI views inside Xcode without launching the app. Hot-reload-ish; useful, but can be flaky.

# Apple's Gatekeeping Infrastructure
- ==Apple Developer Program==: $99/year membership needed to distribute through App Store
- ==[[App Store Connect]]==: Apple's web portal, manages your apple's metadata, screenshots, builds, beta testers, sales, reviews. The "[[Control Plane]]" for everything post-build.
- ==Bundle ID==: Globally-unique [[Reverse DNS]] identifier for your app, e.g. `com.yourco.events`.  Locked once chosen for an app on the store.
- ==Code Signing==: Every binary that runs on iOS must be [[Cryptographic Signature|Signed]] with an Apple-issued cryptographic identity.
- ==Certificates==: Your Apple-issued signing identity. Two main kinds: development and distribution.
- ==Provisioning Profile==: A signed file linking a certificate + a bundle ID + (for dev) a list of allowed device UDIDs + a list of "entitlements" the app is allowed to use. Xcode handles this mostly automatically.
- ==Entitlements==: Capabilities your app declares it needs: push notifications, location-when-in-use, location-always, camera, contacts, HealthKit, in-app purchase, app groups, iCloud, etc. 
	- Each must be declared in your provisioning profile and often in your `Info.plist` with a user-facing `purpose string` explaining why.
- ==Info.plsit==: XML config file describing the app: name, version, required device capabilities, supported orientation, all your purpose strings ("This app uses your location to show nearby events.")
- ==TestFlight==: Apple's beta distribution. Two tracks: internal (up to 100 team members, no review) and external (up to 10k testers via public link/invite, lightweight review)
- ==App Review==: The human + automated gate before App Store release. Driven by the App Store Review Guidelines (a 50+ page document worth skimming).
- ==App Privacy Manifest / Nutrition Labels==: Apple's mandatory disclosure of what data your app collects and why. Shown on the App Store listing.
- ==Phased release==: Opt-in 7-day auto-rollout  (1% -> 2% -> 5% -> 10% -> 20% -> 50% -> 100%). USE IT!


# Apple's Standard Frameworks
- ==Foundation==: Base types (String, DAte, URL, URLSession, JSONDecoder, Data, etc.). Always present.
- ==CoreLocation==: GPS, geofencing, region monitoring, heading. Useful for a check-in radius in a PGo-like app.
- [[MapKit]]: Apple's map framework
- ==UserNotifications==: Local notifications (scheduled by the app) + push notifications (delivered by Apple). You request permission, register, and get a device token to send to your backend.
- ==AVFoundation==: Camera, audio, video capture and playback; Powers things like in-app selfie captures.
- ==Vision==: Apple's on-device computer vision; face detection, face landmarks, OCR, object recognition. Fast, free, and runs offline. Useful for liveness checks.
- ==PhotoUI/PhotoKit==: Photo library access (e.g. for letting users upload event photos)
- ==CoreData / SwiftData:== Local persistence. CoreData is the old (~15-year-old) ORM; SwiftData is the new SiftUI-flavored layer on top. Both are object graph stores backed by SQLite.
- Combine / ==Swift Concurrency==: Combine is Apple's RxJava/RxJS-style reactive framework, while Swift Concurrency is the newer async/await + actors model. Both coexist, but new code defaults to async/await.
- ==WidgetKit, App Intents, BackgroundTasks==: Extension points for home screen widgets, Siri shortcuts, scheduled background refresh. Probably out of scope for v0 of our event app.

# Push Notifications Specifically
- [[Apple Push Notification service]] (APNs): Apple's push server. You don't talk to phones directly; you POST to APNs (HTTP/2 + JWT auth), Apple delivers. Free!
- Device token: Opaque identifier APNs give your app on registration. You send it to your backend so that your your backend can address pushes to that device.
- APNs key (.p8 file): The auth credential your backend uses to talk to APNs. Generated in App Store Connect.
- FCM (Firebase Cloud Messaging): Google's cross-platform push abstraction. Sits on top of APNs for iOS. Lets you target both Android and iOS through one API. Common choice even for iOS-only apps because it bundles with Firebase Analytics.

# Dependency Management
- Swift Package Manager (SPM): Apple's official package manager, integrated into Xcode. Add a Git URL -> done. Equivalent of `npm install`, default choice for new code.
	- CocoaPods and Carthage are older package managers.

# CI/CD and release automation
- [[Fastlane]]: Open-source Ruby toolchain that automates the painful parts: managing certs/profiles (match), uploading to TestFlight (pilot), pushing to App Store (deliver), generating screenshots (snapshot). Basically the "I don't want to wrestle Apple's tooling by hand" layer. Ubiquitous.
- Xcode Cloud: Apple's first-party CI. Integrated, simple, expensive at scale.
- GitHub Actions / Bitrise / Codemagic: Third-party CI services with hosted Mac runners. Common alternatives.

# Common Third-party SDKs
- [[Firebase]]: Google's all-in-one BaaS. Auth, Analytics, Crashlytics, Remote Config ([[Feature Flag]]s + [[AB Testing]]), FCM (push), Realtime Database / Firestore. Free tier is generous.
- [[Supabase]]: Open-source Firebase alternative. Postgres + auth + realtime + storage. Has a Swift SDK.
- [[Sentry]]: Third-party crash + error reporting with much better dev UX than Apple's built-in crash logs.
- [[PostHog]]/Amplitude/Mixpanel: Product analytics (event tracking, funnels, retention cohorts)
- ==RevenueCat==: Handles in-app purchases, subscriptions
- Persona / Onfido / Veriff / Facetec

# Cross-Platform (worth knowing the names, even though we're not using them)
- [[React Native]]: Meta's framework. JS/TS renders to native components. Mature, big ecosystem, custom map/camera work is awkward.
- [[Flutter]]: Google's framework for cross-platform. using the Dart language. Draws its own widgets via Skia.
- Expo: Managed wrapper around [[React Native]] that hides the native toolchain.
- Capacitor / Ionic: Web tech (HTML/CSS/JS) wrapped in a native shell. Fine for content apps.

### Specific to our PGO Events app:
- Mapbox iOS SDK (or MapKit or MapLibre Native): The Map renderer.
- CoreLocation for GPS + the check-in radius enforcement
- AVFoundation + Vision for the selfie capture + lightweight liveness check.
- APNs (likely frontend via FCM or a backend abstraction) for push.
	- FCM = "Firebase Cloud Messaging; Google's push notification service, originally Android-native but now cross=platform."
- A backend SDK of some flavor (Firebase, Supabase, or our own REST/WebSocket client)

























