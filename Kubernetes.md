---
aliases:
  - Kubelet
  - K8s
---
- Pod: Grouping of one or more containers into Pods. A unit that shares the same network and storage, good for tightly-coupled applications, or just a single container
- Deployment: Manages a set of pods to run an application workload, usually one that doesn't maintain state.  Defines how many replicas of a pod should be running, and K8s makes sure that number is maintained. Provides declarative updates for Pods and Replica Sets
- Replica Sets: Ensures a specified number of Pod replicas are running at one time. You describe the desired state in the deployment, and the deployment controller changes the actual state to the desired state at a determined rate. The controller is as loop that watches the shared state of the cluster through the API Server, and makes changes, attempting to move the current state to the desired state.
- Autoscaling:
- Horizontal Pod Autoscaler (HPA)
- Vertical Pod Autoscaler (VPA)
- Cluster Autoscaler
- Service: Load Balancing of incoming network traffic across pods.
- While Docker runs containers, K8S handles deployment, scaling, orchestration, management across hundreds of servers.


