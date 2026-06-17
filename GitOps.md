GitOps is ==an operational framework that applies [[DevOps]] best practices—such as version control, collaboration, and CI/CD—to infrastructure automation==, using Git as the "single source of truth" for declarative infrastructure. It ensures that the entire system state is stored in a repository, allowing automated tools to reconcile the actual infrastructure state with the desired state defined in code, which is ideal for managing [[Kubernetes]].

GitOps means:
> Git contains the desired operational state.
> Automation continuously makes the real system match Git.

In [[Kubernetes]], this means something like [[ArgoCD]] watches a Git repository containing manifests, Helm values, or Kustomize overlays, then reconciles the cluster to match those files. Operations then become declarative, reviewed, versioned, and continuously reconciled, instead of just being a sequence of manual commands.
```
Git repository
	-> Kubernetes manifests / Helm values / Kustomize overlays
	-> ArgoCD or Flux watches the Repo
	-> K8s controller applies the desired state to Kubernetes
	-> cluster converges toward Git
```
The idea is that Git isn't just where code lives, it becomes the source of truth for operational state.
GitOps is Kubernetes’s reconciliation idea applied one level higher.

Without GitOps, deployments often happen by imperative commands:
```
kubectl apply -f deployment.yaml
helm upgrade payments-api ./chart
terraform apply
```
These commands mutate infrastructure directly.