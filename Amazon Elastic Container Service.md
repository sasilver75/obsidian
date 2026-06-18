---
aliases:
  - AWS Elastic Container Service
  - AWS ECS
  - ECS
---
AWS's managed container orchestrator. 

Give AWS a container image, resource requirements, networking rules, and a desired number of running copes, and ECS places and supervises those containers for you.

> A fully managed service for deploying, managing, and scaling containerized applications, without managing a separate orchestration control plane.

Think of it as three layers:
1. Capacity: The compute where containers actually run
2. Scheduler/controller: The ECS service logic that places tasks, replaces failed tasks, and rolls out deployments.
3. Provisioning interface: The AWS console, AWS command line interface, AWS CloudFormation, AWS Cloud Development Kit, Terraform, or SDK calls you use to declare what should exist

Core objects:
- Cluster: A logical grouping of ECS tasks and services, plus the capacity they run on. Clusters are region-specific.
- Task Definition: The JSON blueprint for a runnable unit: Container image, CPU, memory, ports, environment variables, logging, volumes, IAM roles, and network mode.
- Task: A running instantiation of a task definition. A task can contain one or more containers.
- Service: A long-running controller that keeps a desired number of tasks alive, and can attach them to a LB.
- Capacity provider: A strategy object tells ECS what kind of compute to use and how to distribute tasks across compute pools.

Typical Workflow:
1. You push an image, e.g. `api:2026-06-16` to [[Amazon Elastic Container Registry|AWS ECR]]
2. You register an **ECS task** definition that says: "Run this image, allocate 0.5 vCPU and 1GB memory, expose Port 8080, and send logs to [[Amazon CloudWatch|CloudWatch]], and use this task [[Amazon Identity and Access Management|IAM]] role."
3. You create an **ECS service** with desired count `3`
4. ECS starts 3 tasks across available subnets/capacity.
5. ECS registers those tasks with an [[Amazon Application Load Balancer|AWS ALB]] target group
6. If one task crashes, ECS starts a replacement task
7. If you deploy a new task definition revision, ECS gradually starts new tasks and drains old tasks according to your deployment settings.

```
For HTTP services:
User
  -> Route 53 DNS name
  -> Application Load Balancer
  -> ECS service
  -> ECS tasks
  -> containerized application

For queue workers:
Amazon SQS queue
  -> ECS service or scheduled RunTask calls
  -> worker tasks
  -> database / API / downstream system
```

ECS doesn't actually prescribe what compute you use underlying it. You can use [[Amazon Fargate|AWS Fargate]], ECS managed instances, or [[Amazon EC2|EC2]] instances. Current AWS docs emphasize ECS Managed Instances as the recommended option for many new workloads.

# ECS vs [[Amazon Elastic Kubernetes Service|Elastic Kubernetes Service]] (EKS)
- ECS is simpler and more AWS-native, while EKS gives you Kubernetes portability and the Kubernetes ecosystem.
- A common misconception is that "ECS is less serious than Kubernetes." More precisely, ECS is less *general* than Kubernetes, which can be a feature when the system only needs to run production containers on AWS.