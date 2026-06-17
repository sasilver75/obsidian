---
aliases:
  - Kubelet
  - K8s
---
References:
- Video: [TechWorld with Nana's Kubernetes Crash Course](https://youtu.be/s_o8dwzRlu4?si=QIbUaOx7b8uUdu7G)

An open-source container orchestration system that runs containerized applications across a cluster of machines, and continuously tries to keep the real system matching the desired state you need.
- Kubernetes answers: “How do I run many containers across many machines, keep them healthy, update them safely, scale them, connect them, and recover when things fail?”
- Provides:
	- A standard API for deployment
	- Scheduling across machines
	- Service discovery
	- Self-healing
	- Scaling
	- Rollout mechanics
	- Extensibility through custom resources and controllers
	- Portability across cloud providers, at least in theory
- At the cost of complexity.
	- Kubernetes is worth if if you have many services, multiple teams, high availability requirements, frequent deployments, need for autoscaling, platform engineering investment, etc.
	- It's overkill if you have one small app, one database, simple traffic patterns, a small team, low operational complexity, no need for custom orchestration

Say you have a web application packaged as a container image:
```
registry.example.com/payments-api:v42
```
Running one copy with [[Docker]] is easy: `docker run registry.example.com/payments-api:v42`
But production usually needs more than one container, and K8s can help with all of:
- Run 6 copies of the service
- Spread those copies over multiple machines
- Restart failed containers
- Avoid putting all replicas on the same failing machine.
- Send traffic only to healthy replicas.
- Roll out version`v43` gradually, and roll it back if it fails.
- Give the app environment variables, secretes, CPU limits, memory limits, and persistent storage.
- Let other services find it by a stable name like `payments-api`
- Keep doing all of this even when machines disappear.

Kubernetes does this in a declarative way, and behind the scenes can be understood as a ==desired-state machine for infrastructure.==
You don't usually tell K8s: "Start container A on machine 3, then start container B on machine 7," instead you say: "I want 6 healthy replicas of this application, running this container image, listening on this port, with these CPU and memory limits." And then Kubernetes ==continuously compares the desired state to the actual state, and takes actions to converge the actual state to the desired state.== 

# The Main Components of Kubernetes
1. Cluster
2. Container
3. Pod
4. Deployment
5. ReplicaSet
6. Service
7. Ingress
8. Configmap # Sam
9. Secrets # Sam
10. Volumes # Sam
11. StatefulSet # Sam

#### Cluster
- A ==Cluster== is a group of machines managed as one system, where the machines are usually called nodes.
- There are two broad categories of components:
	- ==Control plane==: The components that make decisions about the cluster. Decide what should happen.
	- ==Worker nodes==: The machines where application workloads actually run. Does the work.

#### Container
- A [[Container]] is an isolated process with its own filesystem, environment, network view, and resource limits.
	- Kubernetes usually runs containers using a container runtime that understands [[Open Container Initiative]] (OCI) images. So it doesn't *have* to be the [[Docker]] engine.

#### Pod
- A ==Pod== is the smallest deployable unit in Kubernetes, containing one or more containers that are scheduled together onto the same node. 
- Containers in the same Pod share:
	- The same network namespace
	- The same IP address
	- Usually the same lifecycle
	- Optionally, shared volumes
- Most application Pods contain exactly one container; you'd use multi-container pods if the containers are tightly coupled, such as an application container plus a sidecary proxy, or log collector.
- ==Kubernetes does not manage containers; it manages Pods, which contain container(s).==

#### Deployment
- A ==Deployment== describes the desired state of a stateless application that should keep some number of Pod replicas running:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
spec:
  replicas: 6
  selector:
    matchLabels:
      app: payments-api
  template:
    metadata:
      labels:
        app: payments-api
    spec:
      containers:
        - name: payments-api
          image: registry.example.com/payments-api:v42
          ports:
            - containerPort: 8080
```
This says: "Run 6 pods matching this template. Each pod should match the `payments-api` container image."
- If one Pod crashes, Kubernetes creates a replacement. If a node dies, Kubernetes schedules replacement Pods elsewhere, assuming enough capacity exists.

#### ReplicaSet
- A ==ReplicaSet== ensures that a specified number of matching pods exist.
- You don't create ReplicaSets directly, a Deployment creates and manages ReplicaSets for you!
```
Deployment -> ReplicaSet -> Pods -> Containers
```
The Deployment handles rollouts and rollback history, while the ReplicaSet handles replica count, while the Pods run the containers.

#### Service
- A ==Service== gives a stable network identity to a changing set of Pods.
- Pods are ephemeral: They can be destroyed or replaced and any time, and the replacement Pod usually gets a new IP address. We don't call Pods by their own IP address, this would be too fragile. Instead, we call these Pods via Services, which give a stable name and virtual address to callers.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: payments-api
spec:
  selector:
    app: payments-api
  ports:
    - port: 80
      targetPort: 8080
```
Other services can now call:
```
http://payments-api
```
... and Kubernetes routes that traffic to one of the healthy pods that is selected by `app: payments-api`

This is where Kubernetes overlaps with [[Service Discovery]] and [[Load Balancing]].

#### Ingress
- An ==Ingress== usually exposes HTTP or HTTPS traffic from *outside the cluster* to Services inside the cluster.
- Rough flow:
```
Internet
	-> cloud load balancer
	-> K8s Ingress Controller
	-> K8s Kubernetes Service
	-> K8s Pod
```
- The exact behavior depends on the Ingress controller. For example, [[NGINX]] Ingress, [[Traefik]], [[HAProxy]], [[Istio]], and cloud-provider-specific controllers can all implement Ingress-like behavior.
- Modern Kubernetes also has the ==Gateway API==, which is a *newer, more expressive approach to traffic routing than the older Ingress API.*


#### How K8s work mechanically
The most important mechanism is the control loop:
1. You submit YAML to the Kubernetes ==API Server== via ==Kubectl==
2. The API Server validates and stores the desired state
3. The desired state is persisted in [[etcd]], Kubernete's [[Strong Consistency|Strongly Consistent]] KV store
4. ==Controllers== watch the API Server for changes
5. The ==Scheduler== assigns unscheduled ==Pods== to ==Nodes==
6. The ==Kublet== agent on worker nodes starts and monitors the Pods
7. Kubernetes keeps watching the system's state, comparing it to desired state, and correcting drift

Components:
- API Server: Front door to the cluster; all changes go through it.
- etcd: Stores cluster state.
- Scheduler: Chooses which node should run each Pod.
- Controller Manager: Runs reconciliation loops for Deployments, ReplicaSets, Nodes, Jobs, etc.
- Kubelet: Node agent that starts and monitors Pods.
- Container runtime: Actually runs the containers. Doesn't have to be Docker.
- Kube-proxy/networking layer: Implements Service networking, depending on cluster setup
- Container Network Interface plugin: Provides Pod networking; examples include Cilium, Calico, Flannel.

The key design idea is that ==Kubernetes components mostly communicate by reading and writing state through the API Server. They do not usually coordinate through direct imperative commands.==

Example Failure Case:
- Imagine that you're in a case where you desire that you have 6 replicas of `payments-api`.
- The actual state is also 6 running Pods of `payments-api`. Great!
- Uh oh, one Node dies, taking two Pods with it.
- Now, there are only four running Pods of `payments-api`.
- A Kubernetes ==controller== notices the mismatch and creates two replacement Pods. The ==scheduler== then assigns those ==Pods== to available ==Nodes==. The ==kubelet== agents on those nodes start the containers.
- Clients don't need to know the replacement Pod IPs. They keep calling the Service name.


### Manifests
- A Kubernetes manifest is the file (YAML) that describes K8s API objects such as Deployments, Services, ConfigMaps, Secrets, Ingress, Horizontal Pod Autoscaler, and so on.
> In a system with many application services, you usually do not have exactly “one manifest per service.” You usually have one set of manifests per application service, and that set often contains several Kubernetes objects.

For one application service like `payments-api`, you might have several manifests:

```
payments-api/
  deployment.yaml        # runs the Pods
  service.yaml           # gives the Pods a stable internal network name
  configmap.yaml         # non-sensitive config
  hpa.yaml               # autoscaling rule
  ingress.yaml           # external HTTP routing, if exposed publicly
  serviceaccount.yaml    # workload identity / permissions
```

Alternatively, you might put all of these Kubernetes objects into a single YAML file, separated by `---`:
- Both of these are valid; K8s doesn't care whether thee objects can from one or many files. the file organization is for humans and deployment tooling; the API server just receives API objects.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: payments-api
          image: registry.example.com/payments-api:v42
---
apiVersion: v1
kind: Service
metadata:
  name: payments-api
spec:
  selector:
    app: payments-api
  ports:
    - port: 80
      targetPort: 8080
```

A common production shape is:
```text
k8s/
  payments-api/
    base/
      deployment.yaml
      service.yaml
      hpa.yaml
      kustomization.yaml
    overlays/
      dev/
        kustomization.yaml
      prod/
        kustomization.yaml

  orders-api/
    base/
      deployment.yaml
      service.yaml
      hpa.yaml
      kustomization.yaml
    overlays/
      dev/
        kustomization.yaml
      prod/
        kustomization.yaml
```
Above:
- Q: What is an overlay?
	- A: In the K8s context, it usually means a small set of environment-specific changes layered on top of a shared base configuration. The tool most associated with this pattern is [[Kustomize]]. Without overlays, you often end up duplicating nearly identical YAML. This of an overlay as a patch. Bottom line: An overlay is how you avoid copying entire K8s manifests for every environment.
- The important idea is that K8s is not "the app," it is a declaration of one or more Kubernetes objects that together describes how the app should run. For a typical stateless API, the minimum useful set is.
- So if you have 20 application services, you might have 20 directories, each containing several manifests. Or you might have 20 [[Helm]] charts. Or you might have one monorepo directory with [[Kustomize]] overlays. The exact structure depends on your deployment tooling.
- Common default:
	- ==One application service gets a bundle of manifests: Deployment + Service + config + scaling + routing.==
```
application service
  -> needs several Kubernetes resources
  -> each resource can live in its own manifest file
  -> deployment tooling applies the whole set together
```


### Configuration, Secrets, and Storage
- K8s separates application code from runtime configuration.
There are a variety of common resource types:
- ==ConfigMap==: Non-sensitive configuration, such as feature flags or config files.
- ==Secret==: Sensitive values, such as passwords, API keys, and certificates.
	- Not automatically magically high-security storage! These are K8s API objects intended for sensitive values, but the actual security depends on encryption at rest, access control, secret distribution, node security, and operational practice.
- ==PersistentVolume==: A piece of storage available to the cluster.
- ==PersistentVolumeClaim==: A request by a workload for persistent storage.
- ==StorageClass==: Describes dynamically provisioned storage types.

### Stateless vs Stateful Workloads
- Kubernetes is easiest for stateless services.
- A stateless service can be replaced freely because durable state lives somewhere else, such as PostgreSQL, Redis, S3-compatible object storage, or another external system.
- Stateful workloads are harder, because each replica might need its own stable identity and durable storage.
- Kubernetes uses ==StatefulSets== for workloads like:
	- Databases
	- Distributed queues
	- Consensus systems
	- Systems where each replica has a stable identity
- A StatefulSet provides stable Pod names and stable storage claims:
	- `postgres-0`
	- `postgres-1`
	- `postgres-2`
- However, "can run on Kubernetes" is not the same as "is easy to operate on Kubernetes." Databases on Kubernetes require careful attention to backup, restore, failover, storage performance, upgrade procedure, and data corruption risks.

### Scaling
Kubernetes supports several types of scaling:
1. Manual scaling: Set `replicas:10` yourself.
2. ==Horizontal Pod Autoscaler==: Add or remove *Pod* replicas based on metrics.
3. ==Vertical Pod Autoscaler==: Adjusts CPU and memory requests/limits.
4. ==Cluster Autoscaler==: Adds or removes *Nodes* from the cluster.

### Rollouts and Rollbacks
- Deployments support rolling updates:
	- If you change:
		- `image: registry.example.com/payments-api:v42`
		- to `image: registry.example.com/payments-api:v43`
- Kubernetes can gradually replace old Pods with new Pods:
```
Start some v43 pods
Wait for them to become healthy
Remove some v42 pods
Repeat until all Pods are v43
```
- If the new Pod fails readiness checks, Kubernetes can stop routing traffic to them.
- Depending on configuration/deployment tooling, you may then roll back to the previous version.
- For more advanced strategies, teams often use additional tooling for [[Canary Release]]s, [[Blue-Green Deployment]]s, progressive delivery, or service mesh traffic splitting.

### Health Checks
- K8s supports several health-related probes:
	- ==Startup probe==: "Has this slow-starting app finished starting?"
	- ==Readiness probe==: "Should this Pod receive traffic?"
	- ==Liveness probe==: "Should this container be restarted?"
These probes are NOT interchangeable; It's a common mistake to use a liveness probe for dependency checks... if a database is temporarily unavailable and every application Pod fails its liveness probe, Kubernetes may restart all application Pods, making the outage worse.

### Kubernetes Networking
A few important assumptions:
1. Each Pod gets its own IP address.
2. Pods can usually communicate with other Pods without [[Network Address Translation]], inside the cluster.
3. Services provide stable virtual endpoints over a changing set of Pods.
4. NetworkPolicy can restrict which Pods may talk to which other Pods, if the cluster's networking plugin supports it.

Common traffic path inside the cluster:
```
orderss-api Pod
	-> payments-api Service
	-> one health payments-api Pod
```
Common traffic path from outside the cluster:
```
client
	-> external load balancer
	-> K8s Ingress / Gateway
	-> K8s Service
	-> K8s Pod
```

#### Kubernetes Ecosystem
- [[Helm]]: Package Kubernetes manifests into installable charts
- [[Kustomize]]: Compose and patch YAML without templating
- [[ArgoCD]]/Flux: [[GitOps]] deployment controllers
- [[Prometheus]]: Metrics collection
- [[Grafana]]: Metrics dashboards
- [[OpenTelemetry Protocol|OpenTelemetry]]: Traces, metrics, and logs instrumentation
- [[Istio]]/[[Linkerd]]/[[HashiCorp Consul|Consul]]: Service mesh behavior
- [[cert-manager]]: Automated certificate management
- [[External Secrets Operator]]: Sync secrets from external secret stores
- [[Crossplane]]: Manage cloud infrastructure through Kubernetes-style APIs
- Operators: Custom controllers for managing complex applications


### A Realistic Scenario
Imagine an e-commerce system with these services:
```
frontend
orders-api
paymentes-api
inventory-api
email-worker
```
Each service is first packaged as a container image.
In Kubernetes:
- each API is a ==Deployment==
- each Deployment creates ==Pods==
- each internal API has a ==Service==
- the frontend is exposed through an ==Ingress== or ==Gateway==
- ==ConfigMaps== provides non-sensitive configuration
- ==Secrets== provide database credentials and API tokens
- ==Horizontal Pod Autoscalers== scale busy APIs
- ==readiness probes== prevent broken Pods from receiving traffic
- ==Prometheus== scrapes metrics
- ==Argo CD== keeps the cluster synchronized with Git

In real systems, the hard parts are observability, networking, security, resource limits, storage, deployment safety, and debugging interactions between many independent services.


#### The Kubernetes Object Model
- Remember that K8s is fundamentally an API for objects.
- Most K8s resources follow this shape:
```bash
kubectl get deployment payments-api -o yaml`
```
You'd get back something like
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
spec:
  replicas: 3
status:
  availableReplicas: 3
```
The key split:
- `metadata`: Name, namespace, labels, annotations
- `spec`: Desired state: what you want. You write this in your manifest.
- `status`: Observed state; what K8s sees right now. Kubernetes writes this.

#### Labels and Selectors
- Labels are how Kubernetes objects find each other.
If a Deployment creates Pods with labels:
```yaml
labels:
	app: payments-api
```
Then later, a Service selects Pods with matching labels:
```yaml
selector:
	app: payments-api
```
So Services don't point directly at Deployments; they point at Pods whose labels match the selector. This indirection is powerful, but can cause many bugs! A typo in a label can produce a Service with no backing Pods.

#### Namespaces
- A namespace is a logical partition inside a cluster. (`dev`, `staging`, `prod`, `observability`, `platform`, etc.)
- Namespaces help separate names/permissions/quotas/ownership.
- Namespaces are not hard security boundaries by themselves. Proper isolation also needs role-based access control, network policy, admission controls, and sometimes separate clusters.

#### Resource Requests and Limits
- Every serious K8s user needs to understand CPU and memory requests/limits.
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "1"
    memory: "1Gi"
```

#### Scheduling
The scheduler decides which node should run a Pod.
It considers:
- Resource requests
- Available node capacity
- Node labels
- Taints and tolerations ((?))
- Affinity and anti-affinity rules ((?))
- Topology spread constraints
- Volume placement constraints
This matters, when you want to say something like:
```
Don't put all replicas on the same node.
Run this workload ONLY on GPU nodes.
Keep this service close to that service.
Avoid scheduling batch jobs on latency-sensitive nodes.
```

#### Probes and Pod Lifecycles
A Pod moves through phases like:
```
Pending: K8s accepted the pod, but one or more containers aren't running yet.
Running: The Pod is bound to a node, all containers created, and at least one is running.
Succeeded: All containers terminated successfully and won't be restarted (e.g. Jobs).
Failed: All containers terminated, and at least one container failed/terminated by system.
Unknown: K8s cannot detrmine the Pod state, likely because Node is unreachable.
```
Containers can also be in states like 
```
Waiting
Running
Terminated
```

The major probes are:
- Startup probe: Has the app finished starting?
- Readiness probe: Should the Pod receive traffic?
- Liveness probe: Should K8s restart the container?


#### Config Maps, Secrets, and Environment Injection
Common methods:
- Environment variables
- Mounted files
- Command-line arguments
- Sidecar-injected config
- External secret stores
ConfigMaps are for non-sensitive config. Secrets are for sensitive config, but Kubernetes Secrets still require careful access control and encryption practices.


#### Networking
- A large topic! The important subtopics are:
	- Pod networking: How Pods get IP and talk to eachother
	- Service networking: How stable virtual endpoints route to Pods
	- DNS: How names like `payments-api.default.svc.cluster.local` resolve
	- Ingress / Gateway: How external HTTP traffic enters the cluster
	- NetworkPolicy: How east-west traffic is restricted
	- Container Network Interface plugins: How implementations like Cilium, Calico, or Flannel provide networking.
		- ==Kubernetes defines networking expectations, but a networking plugin actually implements those expectations!==

#### Storage
- Where K8s stops feeling simple:
	- Volume: Storage mounted into a Pod
	- PersistentVolume: Cluster-level storage resource
	- PersistentVolumeClaim: A workload's request for storage
	- StorageClass: A class of dynamically-provisioned storage
	- StatefulSet: Controller for stateful workloads with stable identity/storage

Pods are disposable; Data must often not be disposable.

#### Security
- Security is K8s is not one feature, it's a stack of features:
	- [[Role-Based Access Control|RBAC]]
	- ServiceAccounts
	- Secrets management
	- Admission controllers
	- Pod Security Standards
	- image signing and scanning
	- NetworkPolicy
	- Workload identity
	- Node security
	- Audit logs

> Every workload should have the minimum identity, permissions, network access, and filesystem access it needs.

#### Rollouts, Rollbacks, and Release Strategies
- Rolling updates
- Recreate deployments
- [[Blue-Green Deployment]]s
- [[Canary Release]]s
- Feature flags
- Progressive delivery
- Rollback triggers
- Readiness gates
- Database migration ordering
Kubernetes can restart and replace Pods, but it doesn't automatically make every application-level change safe. A backwards-incompatible DB migration can break even if the K8s rollout itself succeeds.


#### Observability and Debugging
- You need to become fluent with:
```bash
kubectl get pods
kubectl describe pod <pod>
kubectl logs <pod>
kubectl logs -f <pod>
kubectl exec -it <pod> -- sh
kubectl get events
kubectl rollout status deployment/<name>
kubectl rollout undo deployment/<name> 
```


#### Jobs, CronJobs, DaemonSets, StatefulSets
- Deployment: Stateless long-running services
- StatefulSet: Stateful services needing stable identity
- DaemonSet: One pod per node, often by agents
- Job: Run-to-completion task
- CronJob: Scheduled run to-completion task


#### Helm, Kustomize, GitOps
- Raw manifests become painful at scale.
- Common tools:
	- [[Helm]]: Used to package and template Kubernetes resources
	- [[Kustomize]]: Compose and patch YAML variants
	- [[ArgoCD]]: Continuously sync cluster state from Git ([[GitOps]])
		- Flux: Another GitOps controller option


The big [[GitOps]] idea is just: Git contains desired state. A controller continuously reconciles the cluster to match Git.

#### Custom Resources and Operators
- Kubernetes can be extended with *new resource types* that aren't build into core K8s.
	- Careful, these can also increase platform complexity and failure modes.

#### Managed Kubernetes vs Self-Managed Kubernetes
- Most teams should ==NOT RUN THE CONTROL PLANE THEMSELVES== unless they have a strong reason!
- Instead, use:
	- [[Amazon Elastic Kubernetes Service]] (EKS)
	- [[Google Kubernetes Engine]]
	- Azure Kubernetes SErvice
Managed Kubernetes usually handles the control plane, but you still own many hard things:
- Workloads
- Node pools
- Networking choices
- Security posture
- Cluster upgrades
- Observability
- Cost control
- Incident response



# Comparison with [[Terraform]]****
- Terraform usually manages the cloud infrastructure *underneath the application.* Things like [[Virtual Private Cloud|VPC]]s, subnets, [[Amazon Identity and Access Management|IAM]] roles, [[Amazon S3|S3]] buckets, databases, DNS records, [[Amazon EC2|EC2]] instances, and often the K8s cluster itself in the form of things like [[Amazon Elastic Kubernetes Service|EKS]].
- Kubernetes usually manages the application works that *run* on that infrastructure: Pods, Deployments, Services, Ingresses, ConfigMaps, Secrets, autoscalers, and related runtime objects.
- The common stack is: Terraform creates the substrate (network, cluster, node groups, DB, storage, IAM), CI builds and pushes a container image, Kubernetes runs the services, Helm or Kustomize produces/customizes the Kubernetes manifests, and ArgoCD or Flux continuously applies those manifests from Git as GitOps, and Kubernetes rolls out new Pods and keeps them healthy.
- It gets a little fuzzy because Terraform can manage Kubernetes objects, and Kubernetes can indirectly create cloud resources like load balancers or disks... but typically, Terraform is for slower-moving cloud/platform infra, and use Kubernetes plus GitOps tooling for faster-moving application development.
	- In short: Terraform asks “what infrastructure should exist?” while Kubernetes asks “what workloads should be running on this infrastructure?”



______________



kubectl ("Cube Control")
![[Pasted image 20260616151836.png]]
- From your machine, you can connect to the controller node in the cluster. 
- All kubectl requests are received by the ==API server==.
- Text file shown in miniature: A ==kubeconfig== file which is a document that shuld be on the same machine as where `kubectl` is installed. Includes things like:
	- Where is the cluster?
	- Where is the certificate/key file on your personal environment, which lest you authenticate who you are and talk to the cluster.
- The API server is what returns information to you in kubectl, but is also what reaches out and communicates to each node in our cluster, which runs ==kubelet==.
- The API server is the only thing that talks to [[etcd]], which stores all sorts of information about our cluster.

![[Pasted image 20260616152323.png]]
The ==scheduler== is another important part.
- Pods get put on Nodes, but what makes the decision? Kubernetes does, via the Scheduler.
- Tries to make sure that you have an equal sort of distribution across your nodes, plus obeying any rules put into place by you or the system itself.

![[Pasted image 20260616152410.png]]
The ==Controller-manager== is another important part of the control plane.
- Manages many daemons that control behavior of how your cluster runs.
	- Objects in k8s are tracked by namespaces, which can be controlled by [[Role-Based Access Control|RBAC]] which are connected to service accounts...
	- You create copies of pods called replicas, that's something that's tracked as well...
	- ...The controller-manager is in charge of all of this.

==Kubelet== is on every worker node in the cluster, include the controller nodes.
- It's the eyes and ears: Making sure that containers are started, stopped, restarted appropriately in the cluster
- Any instruction coming from the controller node is received by kubelet; if kubelet encounters problems on a node, then your node is now in trouble, because that's your sole connection to the rest of the cluster.

![[Pasted image 20260616152601.png]]
Kubernetes is agnostic in terms of the specific ==container engine== being used (eg podman, docker, etc)
- Inside of these Pods that are being mapped onto nodes are containers.
- What builds those containers? Is it K8s directly? No. Installed on each node in your cluster is some sort of container runtime engine (e.g. Docker).
- So you still need all of the lifecycle things that apply to containers:
	- You still need an ==image==, a read-only template with instructions for containing a container.
	- You still need a ==registry==, which is where images come from.

So how does it happen, when we say: "Make a pod?"
![[Pasted image 20260616152836.png]]
- Kubectl sends some instructions on what K8s is supposed to do
- That information is taken the Controller, specifically to the API server.
	- You can see how much traffic goes to the API server, in this diagram!
- The API server turns to etcd after validating/authorizing your pod plan... which says "YEah, looks good."
- Next, the API server needs to know where the pod is going to go. It goes to the scheduler.
	- Assuming that the manifest didnd't say where the pod should go specifically.
- Scheduler say "Lets put that on Worker Node 2!"
- API server then says "sounds good, I'll let etcd know that we're putting it on node 2", and etcd says "sounds good boss"
- API Server then says back to the scheduler: "Yep, good pick, we're going with Node 2."
- Then our API server reaches out from the Controller Node to the appropriate Worker Node (Node 2)... and sends the instruction to kubelet on Worker Node 2, which says "Got it, boss"
- Kubelet then turns around and runs the necessary container commands (e.g. `docker run ...`)
- Kubelet then constantly starts feeding status updates as the container is built, as it enters an error or stop state, etc... it perpetually provides this information back to the API server, which in turn makes sure that the information is being recorded to etcd.



A ==Manifest== provides a list of descriptions of "These are all the things I want" in a [[Yet Another Markup Langauge|YAML]] file.

The first manifest we'll look at is a Pod:
```yaml
apiVersion: v1
kind: Pod
metadata:
	name: nginx
spec: # This is where you put in your "order" about how you want your pod to be built
	containers:
	- name: nginx
	  image: nginx:1.14.2 # This is where you want to be the most careful!
	  ports:
	  - containerPort: 80

```
- You'll see apiVersion, kind, and metadata inside *all* manifests.
	- apiVersion: Everything in K8s is communicated through APIs... there are many inside a K8s cluster, and all of these different APIs are configured to work with different types of objects. v1 is the one that's proficient at creating pods. It's super important that you get the apiVersion value correct.

Let's start building our first pods!
First, let's learn some fundamental kubectl commands
```bash
# returns a list of whatever object you're asking for
kubectl get <object>
kubectl get pod
> No resources found in default namespace.
```
Let's create a manifest, then, in `podmanifest.yml`, like the one above.
Now let's create a pod from that manifest
```bash
# I would like to apply what is at this file location:
# If this pod doesn't exist, kubectl apply will create it. We're applying what's in the manifest against the cluster
kubectl apply -f podmanifest.yml
# now
kubectl get pods
> NAME READY STATUS RESTARTS AGE
> nginx 0/1  ConatinerCreating 0 6s
# Remember taht we put one container in our pod? Just Nginx?
# We can see that we have 0/1, as a result.
# If we run Kubectl get again... there are zero restarts. When you see a lot of restarts on a pod, that's a big red flag. Something is causing that pod to die, and it needs to get fixed.
```

Next command, let's learn about describe
```bash
# kubectl descrbies brings up detailed information about a resource
# kubectl describe RESOURCE RESOURCE_NAME
kubektl descrribe pod nginx
# Gives a huge output. Good for double-checking that resources have been created.
# At the bottom of the output is a section called Events. If a resoruce is busted or not working, describe should be the first tool to reach for, because you'll often see in the Events section what you goofed up.
```
Let's delete resources
```bash
# kubectl delete RESOURCE RESOURCE_NAME
kubectl delete -f podmanifest.yml
# If we don't have the manifest or don't want to go looking for the manifest...
kubectl delete pod nginx
```

Let's talk about namespaces
```bash
kubectl get pods
> No resources foudn in default namespace

kubectl get namespace
NAME   STATUS   AGE
default Active 23h
kube-node-least Active 23h
kube-public Active 23h
kube-system Active 23h
```

So I'm just not looking in the right place!
```bash
kubectl get pods -n kube-system
# Shows some system-integral pods, which are the ones that hold our clustser together.
```

It's easy to make new namespaces (we'll do it imperatively at the command line)
```bash
# Creat a "demo" namespace
kubectl create ns demo

# Offscreen, we'll hop back in our nginx .yml manifest from earlier.
# If you don't set where the manifest is supposed to go, it goes to the default namespace.
# You can add a namespace key under "metadata" section though, "namespace: demo"
kubectl apply -f podmanifest.yml
> pod/nginx created

kubectl get pod
> No resources found in default namespace

# These pods are inside of the "demo" namespace!
kubectl get pod -n demo
NAME READY STATUS RESTARTS AGE
nginx 1/1  Running 0  16s
```

You need to be sure that when resources are created, you aren't taking more than your fair share of resources (cpu, memory, etc). 
We can create resource quotas, which attach to namespaces and set rules about the limits of what everything in a namespace is allowed to consume.
- Resource quotas attach to namespaces.

YAML to define one looks like:
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
	name: tiny-rq
spec:
	hard:
		cpu: "1" # one core of CPU
		memory: 1Gi # One gig of memory
```
`kubectl apply -f my-tiny-rq.yaml -n demo`
Now you can see that our resource quota has been applied:
```bash
kubectl describe ns demo
# Any pod created in the resource now have to adhere to these resource consumption rules.
```

Let's talk about this `apiVersion` line that's inside each manifest.
- Let's imagine that the day is yesterday, a manifest with `apiVersion: vHonkabonka` worked... and today, we rolled out a new cluster at a later version than the one we had yesterday. Suddenly, your manifests don't work. 'There are no matches for kind "Pod" in version "vHonkabonka"'
- When you're upgrading Kubernetes, you're upgrading its APIs, its ability to recognize, configure, and manage different kinds of resources!
	- If you want to be a real pro at K8s, every time that someone talks about upgrading a cluster, I just want you to groan loudly; no one likes upgrading clusters, it's such a huge pain!


Remember: A K8s cluster is an aggregate of all the nodes that are networked together to create one whole.
You constantly have to keep an eye on how many resources can be consumed, and set reasonable guideposts and guardrails to make sure that wasn't being exceeded.

```bash
kubectl top nodes
> error: Metrics API not available
```
We need to first apply a manifest:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/downlaod/components.yaml
```
This manifest is provided by a Github repo from Kubernetes.... this is going to make a whole bunch of objects. We need to give it a minute to spin up. We just created a bunch of Pods that are gathering/harvesting information about a bunch of pods in our cluster.

```bash
kubectl top nodes
> NAME CPU(cores) CPU% MEMORY(bytes) MEMORY%
> node-1 70m 3% 1258Mi 33%
> node-2 ...
# Cool, it shows what percentage of available CPU and memory has been used on the nodes in our cluster!

# We can also use it for pods
kubectl top pods
> No resources found in current namespace

# Okay, let's look in all namespaces!
kubectl top pods -A
NAMESPACE NAME CPU(cores) MEMORY(byteS)
demo nginx 0m 1Mi
kube-system calico-kube-controllers-6f... 4m 13Mi
kube-system calico-node-wgnd 45m 117Mi
...
```

We're going to continue talking about resource control.
We're going to be editing our pod manifests, and give them more parameters, and make them more inline with the way we want them to behave. We want them to behave themselves when it comes to resource consumption; the containers in our pods need ==resources== to live (we don't want them to starve!) but we don't want them to be pigs, either!

**==REQUESTS==** are parameters that you can set in your manifest that guarantees that your pod has a certain amount of resources. This is good, or otherwise they'll starve! 
==**LIMITS**:== It's not unheard of for containers to go amok and consume limitless amounts of resources. Limits put a hard cap on the number of resources a container can shove in their face.

Let's add to our manifest:
```yaml
apiVersion: v1
kind: Pod
metadata:
	name: demo-pod
spec:
	containers:
	- name: nginx
	  image: nginx:1.14.2
	  resources:
		  requests:
			  cpu: 250m # 1 core is equal to 1000 "millicore"
			  memory: "65M"
		  limits: # limits must be at least the same as requests, or you get an error
			  cpu: 500m
			  memory: "130M"
```
Let's take a look back at our demo namespace from earlier:
- Remember that we put a resource quota on it earlier?

When we run `kubectl apply -f manifest.yml -n demo`
We immediately see the Used resource quotas jump to 250m/65M!

**RESOURCEQUOTAS**: Even when you've restricted your containers via requests/limits, at some point you're going to hit the maximum resources your namespace is allowed to use.
- At a certain point, the club is full-up: go home, new entrants!
- New pods would perhaps not be allowed to be created in our cluster.


==Probes== are basically watchdogs you can put on individual pods to look out for an enforce certain behavior
![[Pasted image 20260617114454.png]]
- When you place a probe on a ***container***... that probe is constantly poking that container with a stick. Every couple of seconds or so, it asks "Hey, are you ok?", and the container replies "OMG, Yes!" and this happens over and over again, forever.
- But there are two kinds of probes
	- ==Liveness Probes==: When the container is probed by the probe and you get no response back at all, or the response comes too late, the probe says "Okay, that's a strike!" After a predetermined number of consecutive strikes, the probe *kills the container!* Because turning it off/on again is a time-honored technique of troubleshooting.
	- ==Readiness Probes==: Also checks to see if a container is responsive... but it doesn't kill a non-responsive container. It's the gentle parent: "Ohh, little Timmy isn't awake yet, everyone, leave Timmy the container alone until he's ready, I don't want any traffic to go to him". That pod is removed from any predetermined settings we have that allow other pods/resources to communicate with it. Putting Readiness Probes makes sure that things are ready before they're being accessed.


We can define them in manifests:
```yaml
apiVersion: v1
kind: Pod
metadata:
	name: sise-lp
spec:
	containers:
	- name: sise
	  image: mhausenblas/simpleservice:0.5.0
	  ports:
	  - containerPort: 9875
	  livenessProbe: # Note that it's put on a container.
		initialDelaySeconds: 2 # How soon after creation do we start probing?
		periodSeconds: 5 # How often thereafter do we probe?
		timeoutSeconds: 1 # How long are we giving the container to respond?
		failureThreshold: 3 # How many consecutive failures before we kill this container?
		httpGet:  # the method of the probe; what's the "stick" the probe is useing to tap?
			path: /health
			port: 9876 # We're sending GET requests to container:9876 /health
```
You can tweak these using all sorts of different parameters, etc... What's super nice is that the difference between writing a liveness probe and readiness probe is virtually nil:
- We could literally change `livenessProbe` to `readinessProbe` and the arguments would still work!
	- Instead of killing the container, it would just turn off traffic to the pod when the failure threshold is reached.

Let's see kubectl run, which is a quick and dirty way of creating a pod without needing a manifest
```bash
# This is going to make a pod called demopod, using the nginx image
kubectl run demopod --image=nginx
> pod/demopod created

# If I use port-forward against a pod, I can choose a local port to connect to the container port inside the nginx container itself: LOCALPORT:CONTAINERPORT
# In the nginx container, port 80 is exposed for HTTP. For local port, it could be any TCP port that's available on your system.
kubectl port-forward demopod 2224:80

# Now, in a differen t terminal, we can:
curl localhost:2224
# And we seee in our pod that we're hitting :80
```

I'm going to use a kubectl exec command to open an interactive terminal inside my pod
```bash
kubectl exec -it demopod -- sh
# Now we're inside our demo contaienr, and we can make changes
mkdir -p var/www
echo "HELP I'M STUCK IN A CONTAINER AND I CAN'T GET OUT!"
```

Say  I have a local file that I want to take and copy into a pod:
```bash
# Take my local nginx.conf file and put it in the demopod pod at the path etc/ngingx/nginx.conf, overwriting any file if it's there.
kubectl cp nginx.conf demopod:etc/nginx/nginx.conf

curl localhost:2224
> HELP I'M STUCK IN A CONTAINER AND I CAN'T GET OUT!

# What if we delete our pod?
kubectl delete pod --all
> pod "demopod" deleted

# Let's recreate it
kubectl run demopod --image=nginx
# But now if we run our same command... we won't see our changes! Contaienrs are stateless; all the changes were made to the old container, they're dead and gone.
```

So how do we take ephemeral things like pods and containers and be able to push configurations to it?




______________

- Pod: Grouping of one or more containers into Pods. A unit that shares the same network and storage, good for tightly-coupled applications, or just a single container
- Deployment: Manages a set of pods to run an application workload, usually one that doesn't maintain state.  Defines how many replicas of a pod should be running, and K8s makes sure that number is maintained. Provides declarative updates for Pods and Replica Sets
- Replica Sets: Ensures a specified number of Pod replicas are running at one time. You describe the desired state in the deployment, and the deployment controller changes the actual state to the desired state at a determined rate. The controller is as loop that watches the shared state of the cluster through the API Server, and makes changes, attempting to move the current state to the desired state.
- Autoscaling:
- Horizontal Pod Autoscaler (HPA)
- Vertical Pod Autoscaler (VPA)
- Cluster Autoscaler
- Service: Load Balancing of incoming network traffic across pods.
- While Docker runs containers, K8S handles deployment, scaling, orchestration, management across hundreds of servers.








