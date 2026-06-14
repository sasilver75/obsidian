---
aliases:
  - AWS Lambda
  - Lambda
  - AWS Lambda@Edge
  - Lambda@Edge
---
AWS's event-driven [[Serverless]]/[[Serverless|Functions as a Service]] compute service: You upload code, configure what events should trigger it, and AWS runs that code for you without you having to manage servers.

> A small backend worker that wakes up when triggered, does a job, and then goes idle.

Normally, to run backend code, you need a compute environment:
- A virtual machine, such as an [[Amazon EC2|EC2]] instance
- A container running on something like [[Amazon Elastic Container Service|ECS]] or [[Kubernetes]]
- A long-running server process listening for requests

Lambda solves the problem of running small-to-medium units of backend logic without provisioning, patching, scaling, or operating the underlying machines yourself.

Commonly used for:
- (some) API backends
- Background jobs
- File processing
- Scheduled tasks
- Data pipeline steps
- Queue consumers
- Webhook handlers
- Glue code between AWS services
- Lightweight automation

A simplified flow:
1. An event happens, such as the user uploading `invoice.pdf` to Amazon S3
2. Amazon S3 sends an event to AWS Lambda
3. AWS Lambda finds the configured Lambda function
4. AWS Lambda starts or reuses an execution environment
5. AWS Lambda passes the event data into the handler function
6. Your code runs
7. Your code may call other services, such as DynamoDB, S3, or external APIs
8. AWS Lambda records logs and metrics in Amazon CloudWatch
9. The execution environment may be reused for later invocations, but you cannot depend on it lasting forever

# How does Lambda work mechanically
- Runtime: The language environment, such as Node.js, Python, Java, Go, .NET
- Handler: The specific function AWS Lambda calls when an event arrives
	- Think: "The front door AWS Lambda enters through"
- Function code: The application logic you write
	- Think: "The whole toolbox"
- Event source: The thing that triggers the functions
- Execution role: The [[Amazon Identity and Access Management|AWS IAM]] role granting permissions
- Invocation: One execution of the function
- Execution environment: The temporary environment where the code runs

```python
def lambda_handler(event, context):
    order_id = event["order_id"]
    result = process_order(order_id)
    return {"ok": True, "result": result}
```
Here, `process_order` is your actual business logic (function code), `lambda_handler` is the AWS-specific entry point (handler).

When you create a new AWS Lambda function, you choose one of two deployment package types:
1. A `.zip` file archive: You provide your code files and dependencies in a Zip file. This is the common default, especially for simple Node.js and Python functions. 
2. A [[Container]] image: You provide a Lambda-compatible container image scored in [[Amazon Elastic Container Registry|AWS ECR]]. Used when you need customer dependencies, native libraries, or a larger/custom runtime environment.

For a normal, simple Lambda function, the setup is usually:
1. Create a Lambda function
2. Choose runtime, such as Python/Node/Java/Go/.NET
3. Choose architecture, usually x86_64 or arm64
4. Choose an execution role, which is the [[Amazon Identity and Access Management|AWS IAM]] role the function runs as.
5. Upload code as a `.zip` file, or edit code directly in the console for simple interpreted-language examples.
6. Configure the handler: `lambda_function.lambda_handler` means: "In the lambda_function.py file, use the lambda_handler function." 
7. Separately, configure what actually invokes the Lambda function: API Gateway, S3, SQS, EventBridge, a manual test event, and so on.

```shell
aws lambda create-function \
  --function-name resize-image \
  --package-type Image \
  --code ImageUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/resize-image:latest \
  --role arn:aws:iam::123456789012:role/lambda-execution-role
```

For a container-image Lambda function, the flow is different:
1. Write your function code
2. Write a `Dockerfile`: Use an AWS Lambda base image or provide the Lambda runtime interface client yourself
3. Build the image locally
4. Push the image to [[Amazon Elastic Container Registry|AWS Elastic Container Registry]]
5. In AWS Lambda, create the lambda function using `PackageType = Image`, telling Lambda that the code is packaged as a container image, not as a `.zip` file.
6. Point AWS Lambda at the image URL

```bash
aws lambda create-function \
  --function-name resize-image \
  --runtime python3.12 \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --role arn:aws:iam::123456789012:role/lambda-execution-role
```

# The [[Cold Start]] problem
- Cold Starts happen when AWS Lambda needs to create a new execution environment before running your function. This can add latency, which matters for latency-sensitive applications like user-facing APIs, but perhaps doesn't matter for background jobs, scheduled tasks, or queue processing.
	- A *warm start* happens when AWS is able to reuse an existing execution environment, which is usually faster.
- When AWS reuses an execution environment, cached data can sometimes survive between invocations, but that reuse is an optimization, not a guarantee. Any sort of durable state should live somewhere external, such as in DynamoDB, S3, RDS, etc.







