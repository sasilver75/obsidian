A lightweight, isolated package for running software, with the follow bundled together:
- application code
- runtime
- libraries
- dependencies
- configuration

This lets the same application run consistently across different environments, whether it's a developer laptop or production cluster. ==A portable box for running software the same way everywhere.==
- Built from an image
- Can be started, stopped, copied, deployed, and scaled quickly.

Importantly:
- A container shares its host [[Operating System]] [[Kernel]] 
- A container is more lightweight than a [[Virtual Machine]]

The [[Open Container Initiative]] (OCI) publishes the main container standards:
- OCI Image Specification: What a container image is (layers, metadata, filesystem contents, config)
- OCI Runtime Specification: How to run a container (setup, namespaces, mounts, env variables, lifecycle)
- OCI Distribution Specification: How container images are pushed to/pulled from registries.


Common options: [[Docker]], [[Podman]], [[Containerd]], [[Kubernetes]]

> Think of it as the difference between a food truck and a restaurant.



