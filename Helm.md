
Helm is a packaging and templating tool for [[Kubernetes]]. 

Kubernetes itself wants concrete YAML objects like Deployments, Services, ConfigMaps, and Ingresses.  Helm lets you define these objects as reusable templates, fill in environment-specific values, and render the final Kubernetes manifests that actually get applied to the cluster.

The central object in Helm is a ==chart==: a package containing templates, default values, metadata, and sometimes dependencies. Installing a chart creates a *release*, which is one installed instance of that chart in a cluster.
- For example, the same `payment-api` chart could be installed once as `payments-api-staging` with 1 replica, and `payments-api-prod` with 8 replicas.

The main benefit of Helm is that it avoids duplicating large amounts of Kubernetes YAML across services and environments. The main cost is indirection: the YAML you read in a chart is not always the *exact* YAML Kubernetes receive, because templates and values must be rendered first.

> Chart + Values -> rendered Kubernetes manifests -> live Kubernetes objects


# Comparison with [[Kustomize]]
- Helm is template-based packaging
	- `chart + values -> rendered manifessts`
	- The source files are often not valid YAML, because templates contain `{{ ... }}`
	- Best for reusable, installable packages... especially for third-party systems like [[Prometheus]], [[Grafana]], ingress controllers, or [[cert-manager]].
- Kustomize is patch-base composition
	- `Base manifests + overlays -> final manifests`
	- Usually valid YAML, because bases are plain Kubernetes manifests
	- Best for environment-specific variants of owned manifests. Use Kustomize when you already have clear Kubernetes manifests, and just need variations like `dev`, `staging`, and `prod`. 
- Many teams combine the two!
	- Helm might render a chart, while Kustomize or a GitOps tool like [[ArgoCD]] or Flux handles environment-specific deployment shape.









