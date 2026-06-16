---
aliases:
  - Kubelet
  - K8s
---
References:
- Video: [TechWorld with Nana's Kubernetes Crash Course](https://youtu.be/s_o8dwzRlu4?si=QIbUaOx7b8uUdu7G)

An open-source container orchestration system that runs containerized applications across a cluster of machines, and continuously tries to keep the real system matching the desired state you need.
- Kubernetes answers: “How do I run many containers across many machines, keep them healthy, update them safely, scale them, connect them, and recover when things fail?”

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


#### Service


#### Ingress










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








