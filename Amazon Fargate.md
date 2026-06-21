---
aliases:
  - AWS Fargate
---
AWS's "serverless containers" compute option. You give AWS a [[Container]] image, CPU and memory requirements, networking settings, and [[Amazon Identity and Access Management|IAM]] permissions, and AWS runs the container without making you provision or manage [[Amazon EC2|EC2]] instances.
- ==In practice, Fargate is used *through* [[Amazon Elastic Container Service|ECS]] or [[Amazon Elastic Kubernetes Service|EKS]]== -- it is the compute capacity for tasks or pods, not a whole application platform by itself.


Sort of the serverless version of [[Amazon Elastic Container Service|ECS]]: "Run this container without making me manage servers."
- With ==ECS Fargate==, ECS schedules schedules ECS tasks onto Fargate.
- With ==EKS Fargate==, EKS schedules Kubernetes pods onto Fargate using Fargate profiles.

Use Fargate when you value operational simplicity more than low-level host control.

The biggest confusion: Fargate does not replace ECS or EKS. Fargate replaces your need to directly manage the worker machines. ECS/EKS decide what should run; Fargate provides where it runs.

# Comparison with [[Amazon Lambda|AWS Lambda]]
- AWS Lambda is AWS's "serverless functions" compute option, where you AWS function code or a Lambda-compatible container image, and Lambda runs that code in response to invocations from events such as API Gateway requests, S3 object uploads, EventBridge schedules, queues, streams, or direct SDK calls.
- So Fargate runs containers as tasks or pods, while Lambda runs functions as *event-driven invocations.*
	- "Run this container without managing servers" vs "Run this handler when an event arrives"
	- Fargate container tasks/pods can run as long as the task/service is meant to run, while standard lambda invocations are short-lived, currently up to 15 minutes.




