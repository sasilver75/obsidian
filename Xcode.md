









# Xcode Projects
- The container file (a bundle that appears as `MyApp.xcodeproj`) that Xcode uses to describe everything about how an iOS app gets built. 
- Inside lives a critical `project.pbxproj`, which lists:
	- Every source file, asset, resource that's part of the app
	- The build targets (e.g. the app itself, its test bundle, app extesnions)
	- Build settings (compiler flags, deployment target, signing identity, bundle ID)
	- Build phases (compile sources, copy resources, link frameworks, run scripts)
	- Schemes (which target to build/run/test, environment config)

Think of it as ==the iOS-world equivalent of a package.json + tsconfig.json + a build manifest, all rolled into one.==
- When you "open a projecti n Xcode, you're opening a `.xcodeproj` bundle."


# [[xcodegen]]
- A third-party command-line tool that generates an Xcode project (`.xcodeproj` or `project.pbxproj`) from a human-written [[Yet Another Markup Langauge|YAML]] or [[JSON]] spec file (typically `project.yml`).
	- You define your targets, sources, dependencies, and build settings declaratively in the spec, run xcodegen, and it produces the `.xcodeproj` bundle.
- In contrast, [[project.pbxproj]] is Xcode's native project description, a plain-text file that lives inside an `.xcodeproj` bundle.
	- Xcode readsn ad writes it directly whenever you add a file, change a build setting, or rename a target through the Xcode GUI.
	- It's verbose, auto-generated, and not designed to be human-edited.
	- Every time two developer add files on different branches, the file changes in overlapping ways
	- The merge conflicts are notoriously ugly.
	- But if you're a single developer, you won't deal with merge conflicts, so the cost of using xcodegen (build time dependency, CI job, give up some Xcode GUI ergonomics), it's not necessarily needed/worth it.


