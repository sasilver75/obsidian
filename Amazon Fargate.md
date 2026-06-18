---
aliases:
  - AWS Fargate
---
Sort of the serverless version of [[Amazon Elastic Container Service|ECS]]: "Run this container without making me manage servers."
- With ==ECS Fargate==, ECS schedules schedules ECS tasks onto Fargate.
- With ==EKS Fargate==, EKS schedules Kubernetes pods onto Fargate using Fargate profiles.

Use Fargate when you value operational simplicity more than low-level host control.

The biggest confusion: Fargate does not replace ECS or EKS. Fargate replaces your need to directly manage the worker machines. ECS/EKS decide what should run; Fargate provides where it runs.