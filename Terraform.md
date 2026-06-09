
An [[Infrastructure as Code]] (IaC) tool used to define, create, and manage cloud and infrastructure resources with configuration files. Terraform is best understood as a ==declarative, stateful infrastructure reconciliation engine==.
- Compare with [[Pulumi]], another IAC code that uses general-purposes language (Python/TypeScript/Go, etc.) 
	- Use Terraform if you want the industry-standard IaC tool. It's easy to be overly-clever with Pulumi.

Instead of manually clicking around in AWS, Azure, Google Cloud, Cloudflare, etc., you instead write your desired state in ==HCL== (Hashicorp Configuration Language):
```
resource "aws_instance" "web" {
	ami = "ami-123456"
	instance_type = "t3.micro"
}
```
Terraform then compares the ==desired state== in your files with the ==real state== and makes the needed changes.

Its normal loop is `write -> plan -> apply`. HashiCorp describes Terraform as managing resources through provider APIs, with providers supplying the platform-specific behavior.

Core ideas and vocabulary:
- ==Configuration==: Fields that describe what you want
- ==Provider==: A plugin for a platform like AWS/Azure/GCP/Cloudflare
- ==State==: Terraform's record of what it manages
- ==Plan==: A preview of changes that Terraform will make
- ==Apply==: Executing the planned changes
- ==Destroy==: Remove managed resources


Terraform is declarative, you say:
```
resource "aws_s3_bucket" "app" {
	bucket = "my-company-app-prod"
}
```
We say: "This Terraform workspace should contain an [[Amazon S3|S3]] bucket represented by the address `aws_s3_bucket.app` with these arguments." 
- Terraform then decides whether that means create, update, replace, destroy, or do nothing.


# Main Moving Parts
- ==Terraform Core== is the CLI/runtime. It parses `.tf` files, evaluates expressions, loads modules, builds the dependency graph, asks providers to read current objects, produces a plan, and walks the graph during apply.
- ==Providers== are plugins, released separately from Terraform itself, that know how to talk to APIs such as AWS, Azure, GitHub, Kubernetes, etc. Every managed resource type comes from a provider; without providers, Terraform cannot manage infrastructure.
- ==Resources== are managed objects. Our `aws_s3_bucket.app` is a resource instance address
	- `aws_s3_bucket` is the provider's resource type
	- `app` is Terraform's local name for it
	- The actual AWS bucket name is an argument, such as `bucket = "..."`
- ==Data Sources== read external information without managing it. For example, a data source might look up an existing VPC, AMI , Route53 zone, or IAM policy document. Terraform usually reads data sources during planning, but may defer them to apply if their inputs are unknown until apply.
- ==State== is Terraform's database. It maps Terraform addresses to real remote object identities, stores cached attributes, remembers dependency metadata, and lets Terraform know what it owns.
	- Terraform requires state because a config address like `aws_s3_bucket.app` must be mapped to a concrete remote object.
- ==Backends== decide where state lives. Local state is a file, whereas remote backends store it elsewhere.
	- Can also support [[Lock|Locking]] so that two runs don't mutate the same state concurrently.
- ==Modules== are reusable collections of Terraform configuration. Every directory is a root module when run directly. Child modules are called with `module` blocks and can come from local paths, registries, or VCS sources.
	- A module is just a directory of `.tf` files. If you run `terraform init`, `terraform plan`, and `terraform apply` from that directory, Terraform treats that directory as the root module.
		- In a root module, it's not written like a reusable child module, because it includes a backend block. Backend configuration belongs in the root module only.
- ==Variables==, ==locals==, and ==outputs== form module interfaces. Variables are inputs, locals are internal computed names/values, and outputs expose selected results to humans, parent modules, automation, or remote state.


# Command Flow

`terraform init`: Terraform initializes the working directory, configures the backend, downloads modules, installs providers, and records selected provider versions in `.terraform.locl.hcl`. It's safe to rerun when configuration changes.

`terraform plan`: Terraform refreshes known remote objects, compares current config to prior state, and proposes actions. First run: state has no `aws_s3_bucket.app`, so the plan says create. Later, if someone disables versioning manually, Terraform should detect drift and plan to re-enable it.

`teraform apply`: Terraform executes the accepted plan. It creates the bucket first, then resources that reference `aws_s3_bucket.app.id`, because those references create implicit graph dependencies. After success, it writes the new state.



# Example: S3 Bucket

A modern S3 bucket configuration often uses several [[Amazon Web Services|AWS]] provider resources, because bucket features like versioning, public access blocking, and encryption are modeled separately:

Below: a root module
```
terraform {
	# This says: use Terraform CLI version 1.6.0 or newer.
	required_version = ">= 1.6.0"
	
	# This declares a dependency on the AWS provider plugin from HashiCorp.
	required_providers {
		aws {
			source = "hashicorp/aws"
			version = "~> 6.0"
		}
	}
	
	# Configures where Terraform stores its state file. This DOES NOT create the state bucket.
	# The bucket must exist. This is the bucket that Terraform uses to remember what it manages.
	backend "s3" {
		bucket = "my-terraform-state-prod"
		key = "apps/storage/terraform.tfstate"
		region = "us-east-1"
		use_lockfile = true
	}
}

# Confiures the AWS provider: Says: "MAke AWS API calls in <this> region"
provider "aws" {
	region = var.aws_region
}

# This declares an input variable "aws_region". If you don't override it, Terraform uses us-east-1.
variable "aws_region" {
	type = string
	default = "us-east-1"
}

# This declares an input variable "bucket_name". No default, so Terraform requires you to provide it.
variable "bucket_name" {
	type = string
}

# This creates an actual S3 bucket. The Terraform address is "aws_s3_bucket.app"
# Its real AWS bucket name comes from the "bucket_name" variable. TF creates a bucket w/ this name.
resource "aws_s3_bucket" "app" {
	bucket = var.bucket_name
	
	tags = {
		ManagedBy = "terraform"
		Service = "app-storage"
	}
}

# Enables versioning on the bucket. "aws_s3_bucket.app.id" creates an implicit dependency:
# Terraform knows it must create aws_s3_bucket.app before it can configure versioning.
resource "aws_s3_bucket_Versioning" "app" {
	bucket = aws_s3_bucket.app.id
	
	versioning_configuration {
		status = "Enabled"
	}
}

# This configures S3 public access blocking for the bucket.
# Again, it depends on the bucket, because it references aws_s3_bucket.app.id
resource "aws_s3_bucket_public_access_block" "app" {
	bucket = aws_s3_bucket.app.id
	
	block_public_acls = true
	block_public_policy = true
	ignore_public_acls = true
	restrict_public_buckets = true
}

# This exposes the bucket name after `terraform apply`
output "bucket_name" {
	value = aws_s3_bucket.app.bucket
}

```
Above: The code defines one Terraform workspace that:
1. Uses the AWS provider
2. Stores Terraform state remotely in an existing S3 bucket
3. Creates an application S3 bucket
4. Enables versioning on that bucket
5. Blocks public access on that bucket
6. Outputs the created bucket name


- When you run `terraform init`, Terraform initializes the S3 backend and downloads the AWS provider.
- When you run `terraform plan -var='bucket_name=my-comapny-app-prod`, Terraform determines what it needs to create:
	- `aws_s3_bucket.app`
	- `aws_s3_bucket_versioning.app`
	- `aws_s3_bucket_public_access_block.app`
- When you run `terraform apply -var='bucket_name=my-company-app-prod'` , Terraform creates the bucket first, then applies versioning and public access blocking, then writes the final mapping into remote state.
















