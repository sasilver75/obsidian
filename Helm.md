
Helm is a packaging and templating tool for [[Kubernetes]]. 

> "A Package Manger for Kubernetes to package YAML files and distribute them in public and private repositories."

Kubernetes itself wants concrete YAML objects like Deployments, Services, ConfigMaps, and Ingresses.  Helm lets you define these objects as reusable templates, fill in environment-specific values, and render the final Kubernetes manifests that actually get applied to the cluster.

The central object in Helm is a ==chart==: a package containing templates, default values, metadata, and sometimes dependencies. Installing a chart creates a *release*, which is one installed instance of that chart in a cluster.
- For example, the same `payment-api` chart could be installed once as `payments-api-staging` with 1 replica, and `payments-api-prod` with 8 replicas.
- A Helm Chart is a bundle of YAML files; create your own Helm Charts, push them to a (public or private) registry, and download/use existing ones that other people have pushed and made available.
	- Things like Postgres, ElasticSearch, Prometheus, etc... that all have complex setup, all have charts already available in a Helm repository. Using a simple `helm install <chartname>`, you can reuse the configuration that someone else (e.g. the company that created a tool) already made.

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




![[Pasted image 20260617101823.png]]
For many of these, the configuration is going to be very similar! 
- Without Helm, you'd have to write multiple YAML files for each of these services, each with their own application version and name defined.
- Using Helm, we can define a common blueprint for all of our microservices, with the dynamic values being replaced by placeholders:
![[Pasted image 20260617101951.png]]
- These {{values}} in the template mean that something is coming from external configurations, typically a separate `values.yaml` file where we define those values that we'll use in the template file. These values can also be set via the command line using the `--set` flag.
- This means that in our Build pipeline, we can replace these values on the fly in CI before deploying them!

Another use case: Same set of applications across different Kubernetes Cluster
![[Pasted image 20260617102107.png]]
Instead of deploying the individual YAML files separately in each cluster, you can package them up to make your own chart that has all the necessary YAML files that a particular deployment needs, and then use them to redeploy the same application in different Kubnernetes cluster environments.

Let's look at an example Helm Chart Structure:
![[Pasted image 20260617102221.png]]
- Chart contains name, version, list of dependencies
- values.yaml is place where all the values are configured for the template files. The default values that you can override later.
	- Can overwrite these:
		- `helm install--values=my-values.yaml <charname>`
- The charts/ dir has chart dependencies inside (if your chart depends on otehr charts)
- The templates/ folder is where the templates are stored
When you execute `helm intsall <chartname>`, the template files are filled with the values from `values.yam` you produce Kubernetes manifest that can then be deployed into Kubernetes.
![[Pasted image 20260617104033.png]]





